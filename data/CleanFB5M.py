
in_path = "/Users/jiecxy/PycharmProjects/ture/data/FB5M.name.txt"
out_path = "/Users/jiecxy/PycharmProjects/ture/data/names.trimmed.5M.txt"

line_count = 0
out_file = open(out_path, "w")
for line in open(in_path, "r"):
    line_count += 1
    splits = line.split('\t')
    if len(splits) != 4:
        raise Exception("Invalid question '{}' at line {}".format(line, line_count))
    text = splits[2].strip()[1:-1]
    if text != "":
        out_file.write(splits[0].strip()[1:-1] + '\t' + splits[1].strip()[1:-1] + '\t' + text + '\n')
out_file.close()