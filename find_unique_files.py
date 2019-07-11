import os


def get_names_in_dir(dir):
    for _, _, names in os.walk(dir):
        return set(names)


def find_unique_names(file_names_1, file_names_2):
    unique_names = set()
    paired_names = set()

    for name in file_names_1:
        if name in file_names_2:
            paired_names.add(name)
        else:
            unique_names.add(name)

    for name in file_names_2:
        if not name in paired_names and not name in unique_names:
            unique_names.add(name)
    
    return unique_names
            


def main(dir1, dir2):
    names1 = get_names_in_dir(dir1)
    names2 = get_names_in_dir(dir2)
    unique_names =  find_unique_names(names1, names2)
    print(unique_names)


if __name__ == '__main__':
    dir1 = ''
    dir2 = ''
    main(dir1, dir2)
