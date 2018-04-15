import pandas
import re
import warnings

def preprocess(csv_file):
    # suppress all warnings
    warnings.simplefilter('ignore')
    # convert to lowercase
    csv_file.description = csv_file.description.str.lower()
    # preprocess
    captions_list = csv_file.description.values.tolist()
    for i in range(len(captions_list)):
        try:
            # padding before all punctuations
            caption = captions_list[i]
            caption = re.sub(r'([.,!?()])', r' \1 ', caption)
            caption = re.sub(r'\s{2,}', ' ', caption)
            # add end symbol
            caption += ' <END>'
            captions_list[i] = caption
        except Exception:
            pass
    # replace with the new column
    new_captions = pandas.DataFrame({'description': captions_list})
    csv_file.description = new_captions.description

    return csv_file