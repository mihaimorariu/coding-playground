"""Implements the solution for the rot-equivalence problem."""

from collections import defaultdict
from typing import List

import logging
import unittest

# ===========================================================================================================
#
# Solution explanation
#
# ===========================================================================================================
#
# 1. Determining rot-equivalence between two strings
#
#    First, let us take a look at what it means for two strings of equal length C, s1 and s2, to be
#    rot-equivalent. For simplicity, let us define s[i] as being a mapping the between uppercase letters
#    set and integers, as follows:
#
#                  s[i] = ord(character at position i) - 65, for every i
#
#    Therefore:
#
#                  s[i] = 0, if the character at position i is "A"
#                  s[i] = 1, if the character at position i is "B"
#
#    etc. Also, let us denote rot-equivalence via the "~" symbol:
#
#                  s1 ~ s2 if s2[i] = s1[i] + k (mod 26), for a constant k in [0, 25] and every i
#
#    where 26 is the length of the alphabet (number of uppercase letters).
#
#    Finally, for a given string s, let us define the "A-aligned string", s_a, as being the
#    rot-equivalent of s that starts with "A":
#
#                  s_a ~ s, s_a[0] = 0
#
#    Example:
#
#                  s = "CDCD" -> s_a = "ABAB"https://en.wikipedia.org/wiki/Caesar_cipher
#
#    We can check whether two strings are rot-equivalent as follows:
#
#    1a. Compute k = |s1[0] - s2[0]| and check whether s2[i] = s1[i] + k (mod 26) for every i > 0.
#
#        This is an easy and straightforward approach, but in our implementation we will be using the
#        next method, for reasons explained at point 2b.
#
#    1b. Check if s1_a = s2_a. In fact, we can even prove that the following relation holds
#        true:
#
#                  s1 ~ s2 <=> s1_a = s2_a
#
#        Proof:
#
#            First, a refresher on modular aritmethic properties: :-)
#
#                      (A + B) mod C = (A mod C + B mod C) mod C
#                      (A - B) mod C = (A mod C - B mod C) mod C
#
#            The proof can be split into two parts:
#
#            * Assuming s1_a = s2_a, the following holds true:
#
#                   s1[i] = s1_a[i] + k1 (mod 26), s1_a[0] = 0, for every i
#                   s2[i] = s2_a[i] + k2 (mod 26), s2_a[0] = 0, for every i
#
#               because s1 ~ s1_a and s2 ~ s2_a (k1 and k2 are two constants in the [0, 25] range).
#               But s1_a[i] = s2_a[i] for every i, so:
#
#                   s2[i] = s1_a[i] + k2     (mod 26)
#                         = s1[i] - k1 + k2  (mod 26), for every i
#
#               which means that s1 ~ s2.
#
#            * Assuming s1 ~ s2, the following holds true:
#
#                   s2[i] = s1[i] + k (mod 26), for a constant k and every i
#
#              But:
#
#                   s1[i] = s1_a[i] + k1 (mod 26), s1_a[0] = 0, for a constant k1 and every i
#                   s2[i] = s2_a[i] + k2 (mod 26), s2_a[0] = 0, for a constant k2 and every i
#
#              since s1 ~ s1_a and s2 ~ s2_a, which means that:
#
#                   s2_a[i] - s1_a[i] = s2[i] - k2 - s1[i] + k1 (mod 26)
#                                     = k + k1 - k2             (mod 26), for every i
#
#              But s2_a[0] = s1_a[0] = 0 (by definition of the A-aligned string), so
#
#                   k + k1 - k2 = 0 (mod 26)
#
#              which means that s1_a[i] = s2_a[i] (mod 26) for every i, so s1_a = s2_a.
#
#    In both cases 1a and 1b, we need to loop over all the characters, so the time complexity is O(C).
#
#
# 2. Determining the rot-equivalence classes
#
# Now that we have a way of determining whether two strings are rot-equivalent, we move on to
# grouping the strings provided as input into rot-equivalence classes.
#
# 2a. An easy and straightforward approach is to use brute-force. A simple implementation would look
#     as follows:
#
#     def find_rot_equivalence_classes(strings: List[str]) -> List[List[str]]:
#         if not strings:
#             return []
#
#         classes = []
#
#         for s in strings:
#             if not s:
#                 logging.warning("Empty string received as input, will skip this!")
#                 continue
#
#             j = -1
#
#             for i, group in enumerate(classes):
#                 # We never add an empty list to classes, so we can safely assume that group[0] exists
#                 if is_rot_equivalent(s, group[0]):
#                     j = i
#                     break
#
#             if j == -1:
#                 classes.append([s1])
#             else:
#                 classes[j].append(s1)
#
#          return classes
#
#     What is the time complexity of this approach? Well, we are looping over the strings, so N steps. For
#     every string strings[i] (with i in [0, N - 1]), we loop up to i times over the partial results, so
#     looping takes O(N^2) time. For every pair of strings s1 and s2, we loop up to C times over the
#     characters (where C = max(len(s) for s in strings)), in order to determine if s1 ~ s2. So, in total,
#     the time complexity is O(CN^2). The space complexity is O(N), since we only need an array to store
#     the results.
#
#     Can we do better? Well, we need to loop at least once over the strings, so it cannot get better than
#     O(N). But we could improve the time it takes to find the rot-equivalence class (from O(CN) to
#     O(C)) by using a different data structure and making use of the fact that s1 ~ s2 <=> s1_a = s2_a, as
#     explained in point 1b. This leads us to the next (and currently implemented) solution:
#
# 2b. Use a dictionary with the A-aligned string as keys and the list of rot-equivalence classes as
#     values. For a given string s, we compute the A-aligned string s_a as follows:
#
#         - Compute k = ord("A") - ord(s[0])
#         - Shift the letters by k positions
#
#     We then access the dictionary at key s_a and insert s into the corresponding list. For dictionaries,
#     lookup is an O(1) operation. The space complexity is also O(N).
#
# ===========================================================================================================

def find_rot_equivalence_classes(strings: List[str]) -> List[List[str]]:
    """Groups the input strings into lists of rot-equivalent strings.

    Args:
        strings: List of input strings

    Returns:
        List[List[str]]: List of lists of rot-equivalent strings
    """
    if not strings:
        return []

    classes = defaultdict(list)

    # Time complexity = O(CN), where N is the number of strings and C is the maximum string length:
    #     - O(N) for looping over the strings
    #     - O(C) for computing an A-aligned string
    #     - O(1) for obtaining the list where the current string should be stored

    for s in strings:
        if not s:
            logging.warning("Empty string received as input, will skip this!")
            continue

        classes[rot(text=s, k=ord("A") - ord(s[0]))].append(s)

    return list(classes.values())


def rot(text: str, k: int) -> str:
    """Encrypts a string using the Caesar cipher method.

    Note:
        For more information about the Casar cipher: https://en.wikipedia.org/wiki/Caesar_cipher

    Args:
        text: String that needs to be encrypted
        k: Number of positions in the alphabet that the letters should be shifted by

    Returns:
        str: Encrypted string
    """
    if not text:
        return ""

    letters = []

    for c in text:
        if not c.isalpha() or not c.isupper():
            raise ValueError("Expected only uppercase letters as input!")

        # Shift the letter and ensure its ASCII code is not lower than the ASCII code of "A"
        ord_c = ord(c) + k + ord("A") % ord("A")

        # Also ensure the ASCII code of the shifted letter is not larger than the ASCII code of "Z"
        ord_delta = ord("Z") - ord("A") + 1
        ord_c = (ord_c - ord("A")) % ord_delta + ord("A")

        letters.append(chr(ord_c))

    encrypted = "".join(c for c in letters)
    return encrypted


class TestRotEquivalentStrings(unittest.TestCase):
    def test_default_input(self):
        actual = find_rot_equivalence_classes(["HELLO", "IFMMP", "WORLD", "URYYB", "ASVPH", "SUN"])
        expected = [["HELLO", "IFMMP", "URYYB"], ["WORLD", "ASVPH"], ["SUN"]]
        self.assertEqual(actual, expected)

    def test_null_input(self):
        actual = find_rot_equivalence_classes([])
        expected = []
        self.assertEqual(actual, expected)

    def test_one_input_only(self):
        actual = find_rot_equivalence_classes(["HELLO"])
        expected = [["HELLO"]]
        self.assertEqual(actual, expected)

    def test_all_rot_equivalent(self):
        actual = find_rot_equivalence_classes(["ABC", "BCD", "CDE"])
        expected = [["ABC", "BCD", "CDE"]]
        self.assertEqual(actual, expected)

    def test_all_not_rot_equivalent(self):
        actual = find_rot_equivalence_classes(["AAA", "BBC", "CCE"])
        expected = [["AAA"], ["BBC"], ["CCE"]]
        self.assertEqual(actual, expected)

    def test_all_different_length(self):
        actual = find_rot_equivalence_classes(["HELLO", "OLA", "KONNICHIWA", "BONJOUR"])
        expected = [["HELLO"], ["OLA"], ["KONNICHIWA"], ["BONJOUR"]]
        self.assertEqual(actual, expected)

    def test_invalid_input(self):
        self.assertRaises(ValueError, find_rot_equivalence_classes, ["Hello!"])

    def test_input_contains_empty_string(self):
        actual = find_rot_equivalence_classes(["HELLO", ""])
        expected = [["HELLO"]]
        self.assertEqual(actual, expected)

    def test_empty_string_only(self):
        actual = find_rot_equivalence_classes([""])
        expected = []
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
