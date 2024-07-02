#!/usr/bin/env python3
#
#===- format_checker.py - Format check&gen-------------------*- python3 -*--===#
#
# Copyright (C) [2022-2025] by Cambricon
#
#===------------------------------------------------------------------------===#

import argparse
import logging
import sys
import os
import re

copyright = "//\n" + "// Copyright (C) [2022-2025] by Cambricon.\n" + "//\n" + "//===" + "-" * 70 + "===//\n"
triton_linalg_path = os.path.abspath(sys.path[0] + "../../../..")


def arg():
    """ Argparser. """
    parser = argparse.ArgumentParser('''
        A format script for checking cc/h/td is following llvm format
        or not. Use --check False to gen path.temp under the given path for new file,
        and --check True in precheckin.''')
    parser.add_argument(
        "--path", help="Location of .h/.cpp/.td file to generate template.")
    parser.add_argument("--desc",
                        default="[Desc]",
                        help="Description for file.")
    parser.add_argument("--check",
                        default=True,
                        help="Gen regex to check file format.")
    return parser.parse_args()


class Header(object):
    """ Object to gen/check header. """

    def __init__(self, path, desc):
        """ Init function with args. """
        # Check path first.
        self.__path = os.path.abspath(path)

        file_language = "C++"
        if path.endswith(".cpp") or path.endswith(".cc"):
            self.__has_macro = False
        elif path.endswith(".h"):
            self.__has_macro = True
        elif path.endswith(".td"):
            self.__has_macro = True
            file_language = "tablegen"
        else:
            logging.info("Skip: format_checker is only for .cpp/.cc/.h/.td")
            logging.info("But met " + self.__path)
            sys.exit(0)
        # Gen header.
        filename = os.path.split(path)[1]
        tail = "-*- " + file_language + " -*-===//\n"
        head = "//===- " + filename + " -"
        # Check file name length.
        if (len(tail) + len(head) + len(desc)) > 80:
            logging.error("desc/filename is too long (header total>80):" +
                          path)
            sys.exit(-1)
        # To gen regex/header to match copyright.
        self.__header = head + desc + tail.rjust(81 - len(head) - len(desc),
                                                 "-") + copyright
        self.__regex = head.replace(".", "\.").replace(
            "-", "\-") + ".*" + tail.replace("*", "\*").replace(
                "-", "\-").replace("+", "\+") + copyright.replace(
                    "(", "\(").replace(")", "\)").replace("[", "\[").replace(
                        "]", "\]").replace("-", "\-")
        if self.__has_macro:
            file_macro = self.__path.replace(triton_linalg_path + "/", "").replace("include/", "").replace(
                "/", "_").replace(".", "_").replace("-", "_").upper()
            self.__macro = "#ifndef " + file_macro + "\n"
            self.__macro += "#define " + file_macro + "\n"
            self.__macro += "#endif // " + file_macro + "\n"

    def gen(self):
        """ To gen path.temp under path. """
        fw = open(self.__path + ".temp", "w")
        fw.write(self.__header + self.__macro)
        fw.close()

    def check(self):
        """ To check header/macro. """
        fo = open(self.__path, "r")
        file_string = fo.read()
        fo.close()
        ptrn_header = re.compile(self.__regex, re.DOTALL)
        h_r = ptrn_header.match(file_string)
        if not h_r:
            logging.error(args.path +
                          " header mismatch, pattern should be like:")
            logging.error("\n" + self.__header)
            sys.exit(-1)

        if self.__has_macro:
            ptrn_macro = re.compile(".*" + self.__macro.replace("\n", "\n.*"),
                                    re.DOTALL)
            m_r = ptrn_macro.match(file_string)
            if not m_r:
                logging.error(args.path +
                              " macro mismatch, pattern should be like:")
                logging.error("\n" + self.__macro)
                sys.exit(-1)


if __name__ == "__main__":
    args = arg()
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(args.path):
        logging.info("Skip: no such file")
        sys.exit(0)
    check = args.check
    h = Header(args.path, args.desc)
    if check == "False":
        h.gen()
    else:
        h.check()
