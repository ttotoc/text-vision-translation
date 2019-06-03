# This script is used to format multiple en-ro translation datasets into the following format:
# {english sequence}{TAB character}{romanian sequence}

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#           corpus
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# import xml.etree.ElementTree as ET
#
# DATASET_CORPUS = "S:\\datasets\\eng-ron_corpus.tmx"
# OUTPUT_PATH = "S:\\processed\\dataset_corpus.txt"
#
# tree = ET.parse(DATASET_CORPUS)
# root = tree.getroot()
#
# with open(file=OUTPUT_PATH, mode="w", encoding="utf-8") as file:
#
#     for tu in root.iter("tu"):
#
#         lang_to_text = {}
#
#         for tuv in tu.iter("tuv"):
#
#             lang = tuv.attrib["{http://www.w3.org/XML/1998/namespace}lang"]
#             lang_to_text[lang] = tuv.find("seg").text
#
#         file.write(f'{lang_to_text["en"]}\t{lang_to_text["ro"]}\n')
#
#
# DATASET_NWS_EN = "S:\\datasets\\nws\\setimes_lexacctrain.en"
# DATASET_NWS_RO = "S:\\datasets\\nws\\setimes_lexacctrain.ro"


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#               nws
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# DATASET_NWS_EN = "S:\\datasets\\nws\\setimes_lexacctrain.en"
# DATASET_NWS_RO = "S:\\datasets\\nws\\setimes_lexacctrain.ro"
# OUTPUT_PATH = "S:\\processed\\dataset_nws.txt"
#
# with open(OUTPUT_PATH, 'w', encoding='utf-8') as file_out:
#
#     with open(DATASET_NWS_EN, encoding="utf-8") as file_en,\
#         open(DATASET_NWS_RO, encoding="utf-8") as file_ro:
#
#         for line_en, line_ro in zip(file_en, file_ro):
#             line_en = line_en.rstrip() + '\t'
#             line_ro = line_ro.rstrip()
#
#             file_out.write(line_en + line_ro + '\n')


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#           manythings
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# DATASET_MANYTHINGS = "S:\\datasets\\manythings_ron.txt"
# OUTPUT_PATH = "S:\\processed\\dataset_manythings.txt"
#
# with open(DATASET_MANYTHINGS, encoding='utf-8') as file,\
#     open(OUTPUT_PATH, 'w', encoding='utf-8') as file_out:
#
#     for line in file:
#         file_out.write(line)

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#       VALIDATE FORMAT
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# check if the final dataset respects the format
DATASET_PATH = "S:\\processed\\dataset_all.txt"

with open(DATASET_PATH, encoding='utf-8') as file:
    for i, line in enumerate(file):
        if line.count('\t') != 1:
            raise ValueError(f'Only one TAB character must be in a single line.({i + 1})')