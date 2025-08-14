from tqdm import tqdm
import sys

def num_lines(filepath):
    with open(filepath, 'r') as fp:
        lines = len(fp.readlines())
    return lines

def read_ssearch36_long(filepath):
    '''
    This function reads an ssearch36 output file and formats it into a more managable tsv file format.
    '''
    L = num_lines(filepath)
    formatted_out=""
    with open(filepath, "r") as file:
        for l in tqdm(range(L)):
            line = file.readline()
            if ">>>" in line:
                current_key = line.split(">>>")[1].split(" - ")[0]
                len1 = int(line.split(" - ")[1].split(" ")[0])
            elif ">>" in line:
                important_stuff = line
                for _ in range(2):
                    new_line = file.readline()
                    important_stuff += new_line
                target_key=important_stuff.split(">>")[1].split(" ")[0]
                len2=int(important_stuff.split("(")[1].split(" ")[0])
                alnlen=int(important_stuff.split("aa overlap")[0].split("in")[-1].strip())
                ident1=float(important_stuff.split("% identity")[0].split(" ")[-1].strip())/100

                eval=float(important_stuff.split("E(")[1].split("\n")[0].split(" ")[-1])
                formatted_out += current_key+"\t"+target_key
                for element in  (ident1, alnlen, len1, len2, eval):
                    formatted_out += "\t"+str(element)
                formatted_out += "\n"
    new_filename=filepath
    new_filename = new_filename+"_formatted"
    with open(new_filename, "w") as file:
        file.write(formatted_out)

if __name__ == "__main__":
    filename=sys.argv[1]
    read_ssearch36_long(filename)
