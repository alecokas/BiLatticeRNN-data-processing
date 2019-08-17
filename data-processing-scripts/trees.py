import re

# Code adapted from ./base/conftools/smoothree-ctm.pl

class Tree:
    thresholds = []
    # dictionaries to store the final values associated with the output values
    y_ave, y_slope = {}, {}
    def __init__(self, fname):
        self.read_tree(fname)
    
    def read_tree(self, fname):
        """ Parse the tree into an array of thresholds and a dictionary of
            average values and slopes and get an internal representation of the tree.
        """
        thresholds = []
        # dictionaries to store the intermediate confidence level output values
        y, y_next = {}, {}

        # regular expression to match the tree lines; first group is < or >,
        # second is threshold, third is y or y_next value and forth is * or not.
        # Lines look like this: 4) confidence < 0.660969 5511  7383 F ( 0.39249 0.60751 )
        tree_re = r"\s*\d+\)\s+confidence\s*([<>])\s*(\d(?:.\d*)?)\s+\d+\s+\d+\s+[FC]\s+\(\s+(\d(?:.\d*)?)\s+\d(?:.\d*)?\s+\)\s*(\*)?"
        root_re = r"\s*1\)\s+root"
        with open(fname, 'r') as f:
            for line in f:
                m = re.match(tree_re, line)
                if m:
                    # will be false if m is None
                    if (m.group(1) == '<'):
                        # if lower threshold line
                        thresholds.append(float(m.group(2)))
                        if m.group(4):
                            # If leaf of tree
                            y[float(m.group(2))] = float(m.group(3))
                    elif m.group(4):
                        # If upper threshold line and leaf of tree
                        y_next[float(m.group(2))] = float(m.group(3))
                elif not re.match(root_re, line):
                    print("Something went wrong with parsing the tree. The regular expression used might be wrong or out of date.")
                    print("Error when parsing this line:")
                    print(line)
                    exit()
        thresholds.append(0.0)
        thresholds.append(1.0)
        y[0.0] = 0.0
        # sort array of threshold values
        thresholds = sorted(thresholds)

        # add y_next entries to y (now know the right positions)
        prev_threshold = thresholds[0]
        for threshold in thresholds[1:]:
            if not threshold in y:
                y[threshold] = y_next[prev_threshold]
            prev_threshold = threshold

        y_ave, y_slope = {}, {} # dictionaries to store the final values associated with the output values
        # find average values:
        prev_threshold = thresholds[0]
        for threshold in thresholds[1:]:
            y_ave[prev_threshold] = (y[threshold] + y[prev_threshold]) / 2
            prev_threshold = threshold
        y_ave[thresholds[-1]] = (y[thresholds[-1]] + 1) / 2
        # find slopes:
        prev_threshold = thresholds[0]
        for threshold in thresholds[1:]:
            y_slope[threshold] = (y_ave[threshold] - y_ave[prev_threshold]) / (threshold - prev_threshold)
            prev_threshold = threshold

        # move to object fields
        self.thresholds = thresholds
        self.y_ave = y_ave
        self.y_slope = y_slope

    # Convert old value to new value
    def conv_value(self, conf_old):
        thresholds = self.thresholds
        y_ave = self.y_ave
        y_slope = self.y_slope

        # find index of smallest threshold value that is higher or equal to the old confidence value
        threshold_index = min([i for i in range(len(thresholds)) if thresholds[i] > conf_old] + [len(thresholds)-1])
        # **in the original file it is higher or equal but that wouldn't work for 0.0**
        threshold = thresholds[threshold_index]
        prev_threshold = thresholds[threshold_index-1]
        # find new value by linear interpolation between threshold y_ave values
        conf_new = y_ave[prev_threshold] + (conf_old-prev_threshold)*y_slope[threshold]
        # make sure <= 1
        conf_new = min(1.0, conf_new)
        return conf_new
