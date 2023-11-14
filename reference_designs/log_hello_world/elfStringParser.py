import argparse
import subprocess
import re
from typing import Dict

def call_unix_proc(cmd:str)->str:
    cmdlist = cmd.split(" ")
    try:
        output = subprocess.check_output(cmdlist, stderr=subprocess.STDOUT)
        return output.decode()
    except subprocess.CalledProcessError as e:
        print(f"ERROR! {cmd} failed \n\n{e.output.decode()}")
        raise e

def _get_ro_offset(ofile:str)->int:
    s = f"readelf -S {ofile}"
    out = call_unix_proc(s)
    pattern = r"\s*\[\s*[0-9]+\]\s*\.rodata\.DMb\.1\s*PROGBITS\s*([0-9a-z]+)"
    match = re.search(pattern, out)
    if match:
        return int(match.group(1),16)
    return int("70A00",16)

def _gen_string_dict(stringsoutput:str, rooffset:int=0)->Dict[int,str]:
    lines = stringsoutput.split("\n")
    result = {}
    first = True
    first_val = 0
    for line in lines:
        l = line.lstrip()
        try:
            hex_num, text = l.split(' ',1)
            if first:
                first_val = int(hex_num, 16)
                result[rooffset] = text
                first=False
            else:
                result[(int(hex_num,16) - first_val) + rooffset] = text
        except:
            pass
    return result

def main():
    parser = argparse.ArgumentParser(description="A utility to extract a json file of all the format strings and corresponding addresses/locations in an AIE design")
    parser.add_argument("--input", required=True, help="Path to the directory where the project was constructed")
    parser.add_argument("--output", default="elfstrings.csv")
    args = parser.parse_args()
    
    ofile = args.input+"/core_0_2.elf"
    strings_cmd = f"strings --radix x -a {ofile}"
    object_strings_str = call_unix_proc(strings_cmd)
    ro_offset = _get_ro_offset(ofile)
    res = _gen_string_dict(object_strings_str, ro_offset)
    with open(args.output, "w") as fp:
        for addr,s in res.items():
            fp.write(f"{addr},{s}\n")


if __name__ == "__main__":
    main()
