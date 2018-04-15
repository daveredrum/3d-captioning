import pandas
import re
import warnings

def preprocess(csv_file):
    # suppress all warnings
    warnings.simplefilter('ignore')
    # convert to lowercase
    csv_file.description = csv_file.description.str.lower()
    # padding before all punctuations
    # it takes some time
    for i in range(len(csv_file.description)):
        try:
            text = csv_file.description.iloc[i]
            text = re.sub('([.,!?()])', r' \1 ', text)
            text = re.sub('\s{2,}', ' ', text)
            csv_file.description.iloc[i] = text
        except Exception:
            pass