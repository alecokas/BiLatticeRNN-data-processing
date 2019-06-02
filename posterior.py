#!/usr/bin/env python3

# Copyright (c) 2007 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III

"""
Word lattices for speech recognition.
Includes routines for loading lattices in Sphinx3 and HTK format,
searching them, and calculating word posterior probabilities.
"""

import gzip
import re
import math
import functools
import numpy as np

LOGZERO = float('-inf')

def cmp(a, b):
    """Compare two numbers."""
    return (a > b) - (a < b)

def logadd(x,y):
    """
    For M{x=log(a)} and M{y=log(b)}, return M{z=log(a+b)}.
    @param x: M{log(a)}
    @type x: float
    @param y: M{log(b)}
    @type y: float
    @return: M{log(a+b)}
    @rtype: float
    """
    if x < y:
        return logadd(y, x)
    if y == LOGZERO:
        return x
    else:
        return x + math.log1p(math.exp(y-x))

def baseword(sym):
    """
    Returns base word (no pronunciation variant) for sym.
    """
    basere2 = re.compile(r"(?:\(\d+\))?$")
    return basere2.sub("", sym)

class Dag(object):
    """
    Directed acyclic graph representation of a phone/word lattice.
    """
    class Node(object):
        """
        Node in a DAG representation of a phone/word lattice.
        @ivar sym: Word corresponding to this node.  All arcs out of
                   this node represent hypothesized instances of this
                   word starting at frame C{entry}.
        @type sym: string
        @ivar entry: Entry frame for this node.
        @type entry: int
        @ivar exits: List of arcs out of this node.
        @type exits: list of Dag.Link
        @ivar entries: List of arcs into this node
        @type entries: list of Dag.Link
        @ivar score: Viterbi (or other) score for this node, used in
                     bestpath calculation.
        @type score: float
        @ivar post: Posterior probability of this node.
        @type post: float
        @ivar prev: Backtrace pointer for this node, used in bestpath
                    calculation.
        @type prev: object
        @ivar fan: Temporary fan-in or fan-out counter used in edge traversal
        @type fan: int
        """
        __slots__ = 'sym', 'entry', 'exits', 'entries', 'score', 'post', 'prev', 'fan'
        def __init__(self, sym, entry):
            self.sym = sym
            self.entry = entry
            self.exits = []
            self.entries = []
            self.score = LOGZERO
            self.post = LOGZERO
            self.prev = None
            self.fan = 0

        def __str__(self):
            return "<Node: %s/%d>" % (self.sym, self.entry)

    class Link(object):
        """
        Link in DAG representation of a phone/word lattice.
        @ivar idx: Index for this link.
        @type idx: int
        @ivar src: Start node for this link.
        @type src: Dag.Node
        @ivar dest: End node for this link.
        @type dst: Dag.Node
        @ivar ascr: Acoustic score for this link.
        @type ascr: float
        @ivar lscr: Best language model score for this link
        @type lscr: float
        @type lback: Best language model backoff mode for this link
        @type lback: int
        @ivar pscr: Dijkstra path score for this link
        @type pscr: float
        @ivar alpha: Joint log-probability of all paths ending in this link
        @type alpha: float
        @ivar beta: Conditional log-probability of all paths following this link
        @type beta: float
        @ivar post: Posterior log-probability of this link
        @type post: float
        @ivar prev: Previous link in best path
        @type prev: Dag.Link
        """
        __slots__ = ('src', 'dest', 'ascr', 'lscr', 'pscr', 'alpha', 'beta',
                     'post', 'lback', 'prev')
        def __init__(self, src, dest, ascr,
                     lscr=LOGZERO, pscr=LOGZERO,
                     alpha=LOGZERO, beta=LOGZERO,
                     post=LOGZERO, lback=0):
            self.src = src
            self.dest = dest
            self.ascr = ascr
            self.lscr = lscr
            self.pscr = pscr
            self.alpha = alpha
            self.beta = beta
            self.post = post
            self.lback = lback
            self.prev = None

        def __str__(self):
            return "<Link: %s/%d => %s/%d P = %f>" % (self.src.sym,
                                                      self.src.entry,
                                                      self.dest.sym,
                                                      self.dest.entry,
                                                      self.post)

    def __init__(self, htk_file=None, frate=100):
        """
        Construct a DAG, optionally loading contents from a file.
        @param frate: Number of frames per second.  This is important
                      when loading HTK word graphs since times in them
                      are specified in decimal.  The default is
                      probably okay.
        @type frate: int
        @param htk_file: HTK SLF format word lattice file to
                         load (optionally).
        @type htk_file: string
        """
        self.frate = frate
        if htk_file != None:
            self.htk2dag(htk_file)

    fieldre = re.compile(r'(\S+)=(?:"((?:[^\\"]+|\\.)*)"|(\S+))')
    def htk2dag(self, htkfile):
        """Read an HTK-format lattice file to populate a DAG."""
        if htkfile.endswith('.gz'): # DUMB
            fh = gzip.open(htkfile, 'rt', encoding='utf-8')
        else:
            fh = open(htkfile)
        self.header = {}
        self.n_frames = 0
        state = 'header'
        # Read everything
        for spam in fh:
            if spam.startswith('#'):
                continue
            fields = dict(map(lambda t: (t[0], t[1] or t[2]),
                              self.fieldre.findall(spam.rstrip())))
            # Number of nodes and links
            if 'N' in fields:
                nnodes = int(fields['N'])
                self.nodes = [None] * nnodes
                nlinks = int(fields['L'])
                self.links = [None] * nlinks
                state = 'items'
            elif 'NODES' in fields:
                nnodes = int(fields['NODES'])
                self.nodes = [None] * nnodes
                nlinks = int(fields['LINKS'])
                self.links = [None] * nlinks
                state = 'items'
            if state == 'header':
                self.header.update(fields)
            else:
                # This is a node
                if 'I' in fields:
                    frame = int(float(fields['t']) * self.frate)
                    node = self.Node(fields['W'], frame)
                    self.nodes[int(fields['I'])] = node
                    if 'p' in fields and float(fields['p']) != 0:
                        node.post = math.log(float(fields['p']))
                    if frame > self.n_frames:
                        self.n_frames = frame
                # This is a link
                elif 'J' in fields:
                    # Link up existing nodes
                    fromnode = int(fields['S'])
                    tonode = int(fields['E'])
                    ascr = float(fields.get('a', 0))
                    lscr = float(fields.get('n', fields.get('l', 1.0)))
                    link = self.Link(fromnode, tonode, ascr, lscr)
                    self.links[int(fields['J'])] = link
                    if 'p' in fields and float(fields['p']) != 0:
                        link.post = math.log(float(fields['p']))
                    self.nodes[int(fromnode)].exits.append(link)

        # FIXME: Not sure if the first and last nodes are always the start and end?
        if 'start' in self.header:
            self.start = self.nodes[int(self.header['start'])]
        else:
            self.start = self.nodes[0]
        if 'end' in self.header:
            self.end = self.nodes[int(self.header['end'])]
        else:
            self.end = self.nodes[-1]
        # Snap links to nodes to point to the objects themselves
        self.snap_links()
        # Sort nodes to be in time order
        self.sort_nodes_forward()

    def snap_links(self):
        for n in self.nodes:
            for x in n.exits:
                x.src = self.nodes[int(x.src)]
                x.dest = self.nodes[int(x.dest)]
                x.dest.entries.append(x)

    def sort_nodes_forward(self):
        # Sort nodes by starting point
        self.nodes.sort(key=functools.cmp_to_key(lambda x, y: cmp(x.entry, y.entry)))
        # Sort edges by ending point
        for n in self.nodes:
            n.exits.sort(key=functools.cmp_to_key(lambda x, y: cmp(x.dest.entry, y.dest.entry)))

    def n_nodes(self):
        """
        Return the number of nodes in the DAG
        @return: Number of nodes in the DAG
        @rtype: int
        """
        return len(self.nodes)

    def n_edges(self):
        """
        Return the number of edges in the DAG
        @return: Number of edges in the DAG
        @rtype: int
        """
        return sum([len(n.exits) for n in self.nodes])

    def edges(self, ordered=True):
        """
        Return an iterator over all edges in the DAG
        """
        if ordered:
            for x in self.links:
                yield x
        else:
            for n in self.nodes:
                for x in n.exits:
                    yield x

    def traverse_edges_topo(self, start=None, end=None):
        """
        Traverse edges in topological order (ensuring that all
        predecessors to a given edge have been traversed before that
        edge).
        """
        for w in self.nodes:
            w.fan = 0
        for x in self.edges():
            x.dest.fan += 1
        if start == None:
            start = self.start
        if end == None:
            end = self.end
        # Agenda of closed edges
        Q = start.exits[:]
        while Q:
            e = Q[0]
            del Q[0]
            yield e
            e.dest.fan -= 1
            if e.dest.fan == 0:
                if e.dest == end:
                    break
                Q.extend(e.dest.exits)

    def reverse_edges_topo(self, start=None, end=None):
        """
        Traverse edges in reverse topological order (ensuring that all
        successors to a given edge have been traversed before that
        edge).
        """
        for w in self.nodes:
            w.fan = 0
        for x in self.edges():
            x.src.fan += 1
        if start == None:
            start = self.start
        if end == None:
            end = self.end
        # Agenda of closed edges
        Q = end.entries[:]
        while Q:
            e = Q[0]
            del Q[0]
            yield e
            e.src.fan -= 1
            if e.src.fan == 0:
                if e.src == start:
                    break
                Q.extend(e.src.entries)

    def forward(self, lw, aw):
        """
        Compute forward variable for all arcs in the lattice.
        """
        for wx in self.traverse_edges_topo():
            # If wx.src has no predecessors the previous alpha is 1.0
            if len(wx.src.entries) == 0:
                alpha = 0
            else:
                alpha = LOGZERO
                # For each predecessor node to wx.src
                for vx in wx.src.entries:
                    # Accumulate alpha for this arc
                    alpha = logadd(alpha, vx.alpha)
            wx.alpha = alpha + wx.ascr * aw + wx.lscr * lw

    def backward(self, lw, aw):
        """
        Compute backward variable for all arcs in the lattice.
        """
        for vx in self.reverse_edges_topo():
            # Beta for arcs into </s> = 1.0
            if len(vx.dest.exits) == 0:
                beta = 0
            else:
                beta = LOGZERO
                # For each outgoing arc from vx.dest
                for wx in vx.dest.exits:
                    # Accumulate beta for this arc
                    beta = logadd(beta, wx.beta)
            # Update beta for this arc
            vx.beta = beta + vx.ascr * aw + vx.lscr * lw

    def posterior(self, lw=1.0, aw=1.0):
        """
        Compute arc posterior probabilities.
        """
        # Clear alphas, betas, and posteriors
        for w in self.nodes:
            for wx in w.exits:
                wx.alpha = wx.beta = wx.post = LOGZERO
        # Run forward and backward
        self.forward(lw, aw)
        self.backward(lw, aw)
        # Sum over alpha for arcs entering the end node to get normalizer
        fwd_norm = LOGZERO
        for vx in self.end.entries:
            fwd_norm = logadd(fwd_norm, vx.alpha)
        bwd_norm = LOGZERO
        for wx in self.start.exits:
            bwd_norm = logadd(bwd_norm, wx.beta)
        # Check relative difference
        if (fwd_norm - bwd_norm) / bwd_norm > 0.01:
            print("Warning: Forward %.8f disagrees with Backward %.8f" %(fwd_norm, bwd_norm))
        # Iterate over all arcs and normalize
        for w in self.nodes:
            w.post = LOGZERO
            for wx in w.exits:
                wx.post = wx.alpha + wx.beta \
                          - fwd_norm - (wx.ascr * aw + wx.lscr * lw)
                w.post = logadd(w.post, wx.post)

    def arc_posterior(self, aw, lw):
        """Get arc posterior vector."""
        self.posterior(lw=lw, aw=aw)
        posterior = []
        for x in self.edges(ordered=True):
            posterior.append(x.post)
        posterior = np.exp(np.array(posterior))
        return posterior
