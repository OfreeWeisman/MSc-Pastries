"""
Extract features from LALAS
"""

import re

# Functions

def get_n_branching_points(lala):
    # count ) in lala
    n_branch = lala.count(')')
    return n_branch

def _get_L_list(lala):
    # generate the list of L subunits
    # split at a, A, and )
    # remove ( in the remaining strings
    l_list = lala.replace('a',',').replace('A',',').replace(')',',').split(',')
    l_list = [i.replace('(','') for i in l_list if 'L' in i]
    return l_list

def get_longest_L(lala):
    # get the list of Ls, find the length of the longest string in the list
    l_list = _get_L_list(lala)
    if l_list == []:
        longest_l = 0
    else:
        longest_l = max(len(i) for i in l_list)
    return longest_l

def get_longest_L_degeneracy(lala):
    # get the longest L value and count the number of occurrences in the string
    longest_l = get_longest_L(lala)
    degeneracy = lala.count('L'*longest_l)
    return degeneracy

def get_second_longest_L(lala):
    # get the longest L and remove all occurrences in the string
    # get the longest L again
    longest_l = get_longest_L(lala)
    lala_temp = lala.replace('L'*longest_l,'')
    second_longest_l = get_longest_L(lala_temp)
    return second_longest_l

def get_n_LAL(lala):
    # count (overlapping) LAL and LaL motifs in string
    n_lal = len(re.findall('(?=LAL)', lala))
    n_lal += len(re.findall('(?=LaL)', lala))
    return n_lal

def _get_n_tricyclic_subunits(lala):
    # get the number of tricyclic subunits
    # (not the same as number of rings)
    lala_reduced = lala.replace('(','')
    return len(lala_reduced)

def get_ratio(lala, character, case_sensitive = True):
    # get ratio of character w.r.t. number of tricyclic subunits
    n_tri_subunits = _get_n_tricyclic_subunits(lala)
    if case_sensitive:
        count_character = lala.count(character)
    else:
        count_character = lala.lower().count(character.lower())
    return count_character / n_tri_subunits

def get_a_ratio(lala):
    # get ratio of a/A or A/a, depending on which is larger
    count_a = lala.count('a')
    count_A = lala.count('A')
    if count_a == 0 and count_A == 0:
        ratio = 0
    elif count_A < count_a or count_A == count_a:
        ratio = count_A / count_a
    elif count_a < count_A:
        ratio = count_a / count_A
    
    return ratio

def get_n_rings(lala):
    # get number of rings
    # number of tricyclic units and branches
    # + 2 for terminal rings
    lala_reduced = lala.replace('(','')
    return len(lala_reduced) + 2

def _get_A_list_old(lala):
    # generate the list of A subunits
    # split at L
    # remove ( and ) in the remaining strings
    a_list = lala.split('L')
    a_list = [i.replace('(','').replace(')','') for i in a_list if 'a' in i or 'A' in i]
    return a_list

def _get_A_list(lala):

    def _get_a_strings(string):
        # intialize list for strings
        string_list = []
        # initalize main branch string
        main_string = ""
        # Toggles on character reading
        read_char = True
        # Counts open branches being read
        branch_counter = 0

        for i,c in enumerate(string):

            # if the character is A/a and reading is toggled, append character to main string
            if c.lower() == 'a' and read_char:
                main_string += c
            
            # if the character is "(", run this function on the remaining string to get the branch string
            elif c == '(':
                # increase the branch counter
                branch_counter += 1
                if read_char:
                    # get the strings in the branch
                    branch_strings = _get_a_strings(string[i+1:])
                    # if the branch is not empty or not just "x", append strings into string_list
                    if not branch_strings == [''] and not branch_strings == ['x']:
                        for branch_string in branch_strings:
                            # if there's no "x" in string (i.e., string is not after L)
                            # add the branch string to the main string before appending
                            if 'x' not in branch_string:
                                string_list.append(main_string + branch_string)
                            # if the string appears after L (x), append the string to string_list
                            else:
                                string_list.append(branch_string)
                    # turn off reading characters inside the branch
                    # (so we don't count the same things twice)
                    # (because the strings inside the branch are already counted with _get_a_strings above)
                    read_char = False

            # if the character is ")"
            elif c == ')':
                # if the branch counter is 0, break the loop
                # (because we have already reached the end of the branch)
                if branch_counter == 0:
                    break
                # otherwise reduce the branch_counter by 1
                else:
                    branch_counter -= 1
                    # if the branch_counter becomes 0
                    # turn on read_char
                    # (which lets the rest of the code know it should check special behavior after branch)
                    if branch_counter == 0:
                        read_char = True

            # if the character is L/l and read_char is on
            elif c.lower() == 'l' and read_char:
                # if the main string is not empty nor x
                if not main_string == "" and not main_string == "x":
                    # append the main string to string list
                    string_list.append(main_string)
                    # and tag the next string as being after L (x)
                main_string = "x"
        


        # if the main string is not empty and not x after the loop, append it to string_list
        if not main_string == "" and not main_string == "x":
            string_list.append(main_string)

        return string_list

    a_list = _get_a_strings(lala)
    a_list = [a.replace('x','') for a in a_list]

    return a_list


def get_longest_A(lala, case_sensitive = True):
    # get the list of As, find the length of the longest string in the list
    # if case_sensitive, split list first wherever A and a next to each other
    a_list = _get_A_list(lala)
    if case_sensitive:
        a_list = [a.replace('Aa','A,a').replace('aA','a,A').split(',') for a in a_list]
        a_list = [i for a in a_list for i in a]
    if a_list == []:
        longest_a = 0
    else:
        longest_a = max(len(i) for i in a_list)
    return longest_a


