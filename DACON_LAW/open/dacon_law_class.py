# !pip install pandas
# !pip install torch
# !pip install tqdm
# !pip install pytorch_transformers
# !pip install transformers
# !pip install catboost
# ! pip install spacy


import numpy as np
import pandas as pd
import re

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from xgboost import XGBClassifier as xgb
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



def text_processor(s):
    """
    문장을 담고있는 variable을 넣어주면
    알파벳을 제외한 문장의 모든 기호, 숫자를 제거합니다.

    :param s: 문장을 담고있는 variable
    :return: 새로운 DataFrame안에 담긴 text_processor가 적용된 column
    """

    pattern = r'\([^)]*\)'  # ()
    s = re.sub(pattern=pattern, repl='', string=s)
    pattern = r'\[[^)]*\]'  # []
    s = re.sub(pattern=pattern, repl='', string=s)
    pattern = r'\<[^)]*\>'  # <>
    s = re.sub(pattern=pattern, repl='', string=s)
    pattern = r'\{[^)]*\}'  # {}
    s = re.sub(pattern=pattern, repl='', string=s)


    pattern = r'[^a-zA-Z]'
    s = re.sub(pattern=pattern, repl=' ', string=s)

    months = ['on january', 'on february', 'on march', 'on april', 'on may', 'on june', 'on july', 'on august', 'on september', 'on october', 'on november', 'on december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for month in months:
        s = s.lower()
        s = s.replace(month, '')


    units = ['mm', 'cm', 'km', 'ml', 'kg', 'g', 'th', 'st', 'rd', 'nd']
    for unit in units:
        s = s.lower()
        s = s.replace(unit, '')

    s_split = s.split()

    s_list = []
    for word in s_split:
        if len(word) != 1:
            s_list.append(word)

    s_list = " ".join(s_list)

    return s_list

def text_processor_2(s):
    """
    문장을 담고있는 variable을 넣어주면
    알파벳을 제외한 문장의 모든 기호, 숫자를 제거합니다.

    :param s: 문장을 담고있는 variable
    :return: 새로운 DataFrame안에 담긴 text_processor가 적용된 column
    """

    pattern = r'\([^)]*\)'  # ()
    s = re.sub(pattern=pattern, repl='', string=s)
    pattern = r'\[[^)]*\]'  # []
    s = re.sub(pattern=pattern, repl='', string=s)
    pattern = r'\<[^)]*\>'  # <>
    s = re.sub(pattern=pattern, repl='', string=s)
    pattern = r'\{[^)]*\}'  # {}
    s = re.sub(pattern=pattern, repl='', string=s)

    pattern = r'[^a-zA-Z0-9]'
    s = re.sub(pattern=pattern, repl=' ', string=s)

    useless = ['et al', 'and', 'inc']
    for word in useless:
        s = s.lower()
        s = s.replace(word, '')

    s_split = s.split()

    s_list = []
    for word in s_split:
        if len(word) != 1:
            s_list.append(word)

    s_list = " ".join(s_list)

    return s_list

def alpha_only_3_cols(df, column1, column2, column3):
    '''
    입력한 df의 column 3개에서 알파벳을 제외한 모든 숫자, 기호를 제거합니다.

    :param df: 대상이 될 DataFrame
    :param column1: df에서 대상이 될 Column 1
    :param column2: df에서 대상이 될 Column 2
    :param column3: df에서 대상이 될 Column 3
    :return: 새로운 DataFrame안에 담긴 text_processor가 적용된 column
    '''

    temp1 = []
    temp2 = []
    temp3 = []
    for i in range(len(df)):
        temp1.append(text_processor(df[f'{column1}'][i]))
        temp2.append(text_processor(df[f'{column2}'][i]))
        temp3.append(text_processor(df[f'{column3}'][i]))
    temp = pd.DataFrame({f"{column1}": temp1, f'{column2}': temp2, f'{column3}': temp3})
    df[f"{column1}"] = temp[f"{column1}"]
    df[f"{column2}"] = temp[f"{column2}"]
    df[f"{column3}"] = temp[f"{column3}"]

    return df

def alpha_numeric_3_cols(df, column1, column2, column3):
    '''
    입력한 df의 column 3개에서 알파벳을 제외한 모든 숫자, 기호를 제거합니다.

    :param df: 대상이 될 DataFrame
    :param column1: df에서 대상이 될 Column 1
    :param column2: df에서 대상이 될 Column 2
    :param column3: df에서 대상이 될 Column 3
    :return: 새로운 DataFrame안에 담긴 text_processor가 적용된 column
    '''

    temp1 = []
    temp2 = []
    temp3 = []
    for i in range(len(df)):
        temp1.append(text_processor_2(df[f'{column1}'][i]))
        temp2.append(text_processor_2(df[f'{column2}'][i]))
        temp3.append(text_processor_2(df[f'{column3}'][i]))
    temp = pd.DataFrame({f"{column1}": temp1, f'{column2}': temp2, f'{column3}': temp3})
    df[f"{column1}"] = temp[f"{column1}"]
    df[f"{column2}"] = temp[f"{column2}"]
    df[f"{column3}"] = temp[f"{column3}"]

    return df


def mean_pooling(model_output, attention_mask):
    '''
    하단 tokenizer를 위한 definition
    '''
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-uncased-contracts")
# model = AutoModelForTokenClassification.from_pretrained("nlpaueb/bert-base-uncased-contracts").to(device)


def auto_tokenizer(df, column_name):
    '''
    입력한 df의 문자 벡터를 수치화 합니다.

    :param df:문자 벡터를 수치화하고 DataFrame
    :return:
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = 'nlpaueb/bert-base-uncased-contracts'
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModelForTokenClassification.from_pretrained(bert_model)
    model = model.to(device)
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, device=0)

    ei_total_list = []
    for text in tqdm(df[column_name]):
        text = text.lower()
        entities = nlp(text)

        party_names = {}
        for entity in entities:
            if 'entity_group' in entity and entity['entity_group'] == 'LABEL_1':
                if 'word' in entity:
                    party = entity['word']
                    if party not in party_names:
                        party_names[party] = {'first_name': '', 'family_name': ''}
                        names = re.findall(r'\b\w+\b', party)
                        if len(names) == 2:
                            party_names[party]['first_name'] = names[0]
                            party_names[party]['family_name'] = names[1]
                        elif len(names) == 1:
                            party_names[party]['first_name'] = names[0]
            else:
                if 'party' in entity:
                    party = entity['party']
                    if party not in party_names:
                        party_names[party] = {'first_name': '', 'family_name': ''}
                    if 'first_name' in entity:
                        party_names['party']['first_name'] = entity['first_name']
                    if 'family_name' in entity:
                        party_names[party]['family_name'] = entity['family_name']
        list_of_states = [
            'wyoming', 'wisconsin', 'west virginia', 'washington', 'virginia',
            'vermont', 'utah', 'texas', 'tennessee', 'south dakota',
            'south carolina', 'rhode island', 'pennsylvania', 'oregon', 'oklahoma',
            'ohio', 'north dakota', 'north carolina', 'new york', 'new mexico',
            'new jersey', 'new hampshire', 'nevada', 'nebraska', 'montana',
            'missouri', 'mississippi', 'minnesota', 'michigan', 'massachusetts',
            'maryland', 'maine', 'louisiana', 'kentucky', 'kansas',
            'iowa', 'indiana', 'illinois', 'idaho', 'hawaii',
            'georgia', 'florida', 'delaware', 'connecticut', 'colorado',
            'california', 'arkansas', 'arizona', 'alaska', 'alabama'

        ]

        list_of_usa = ['usa', 'america', 'u.s.', 'united states', 'the states', 'the us', 'the united states',
                       'the united states of america', 'the u.s.', 'the usa']

        masked_text = text
        for party, names in party_names.items():
            first_name = names['first_name']
            family_name = names['family_name']

            if first_name in list_of_states:
                first_name = '[MASK]'

            if family_name in list_of_states:
                family_name = '[MASK]'

            if first_name in list_of_usa:
                first_name = '[MASK]'
            if family_name in list_of_usa:
                family_name = '[MASK]'

            masked_text = masked_text.replace(first_name, '[MASK]')
            masked_text = masked_text.replace(family_name, '[MASK]')

        for state in list_of_states:
            masked_text = masked_text.replace(state, '[MASK]')

        for usa in list_of_usa:
            masked_text = masked_text.replace(usa, '[MASK]')

        encoded_input = tokenizer(masked_text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
        # encoded_input['input_ids'] = encoded_input['input_ids'].to('cpu')
        # encoded_input['attention_mask'] = encoded_input['attention_mask'].to('cpu')

        with torch.no_grad():

            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        ei_total_list.append(sentence_embeddings.squeeze().cpu().numpy())
    df_berted = np.array(ei_total_list)
    return df_berted


def rename_tokenized(df_1, df_2, column_1, column_2, column_3, target_column):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    df_1_df = pd.DataFrame()
    df_2_df = pd.DataFrame()
    df_list = [df_1, df_2]
    column_list = [column_1, column_2, column_3]
    count = 1
    for df_idx in range(len(df_list)):
        for col_idx in range(len(column_list)):
            df_berted = auto_tokenizer(df_list[df_idx], column_list[col_idx])
            print('\nif statements start\n')
            if isinstance(df_berted, pd.DataFrame):
                df_berted = df_berted.rename(columns={0: f'{column_list[col_idx]}_berted'})
            if df_idx == 0:
                if isinstance(df_1_df, pd.DataFrame):
                    df_1_df = torch.tensor(df_berted, device=device)
                else:
                    df_1_df = torch.cat([df_1_df, torch.tensor(df_berted, device=device)], dim=1)
            elif df_idx == 1:
                if isinstance(df_2_df, pd.DataFrame):
                    df_2_df = torch.tensor(df_berted, device=device)
                else:
                    df_2_df = torch.cat([df_2_df, torch.tensor(df_berted, device=device)], dim=1)
        print(f'If finished, Outer loop in rename_tokenize: {count}')
        count += 1

    df_1_df = df_1_df.cpu()
    df_2_df = df_2_df.cpu()
    df_1_df = pd.DataFrame({'temp' : df_1_df}, index=[0,len(df_1_df)])

    df_1_df = new_tensor_separator(df_1_df, 'temp')
    df_1_df = pd.concat([df_1_df, df_1[target_column]], axis=1)



    print('\nDONE : df_1_df = pd.concat([df_1_df, df_1[target_column]], axis=1)\n')
    df_1_df = df_1_df.cpu()
    df_2_df = df_2_df.cpu()
    return df_1_df, df_2_df


def tensor_2_2d(df, n):
    df_renamed = df.rename(columns={0: 'tbd', 1: 'hmm'})
    tensors = pd.DataFrame(df_renamed.groupby(by="tbd"))
    tensors1 = tensors[1]
    tensors1_df = pd.DataFrame(tensors1)
    tensors1_1 = pd.DataFrame(tensors1_df[1][n])
    target_name_temp = tensors1_1['tbd']
    target = tensors1_1['hmm']
    target_name_df = pd.DataFrame(target_name_temp)
    target_name = target_name_df.iat[0, 0]
    target_df = pd.DataFrame(target)
    target_df = target_df.reset_index()
    target_df = target_df.drop(columns='index')
    target_final_df = target_df.rename(columns={'hmm': target_name})

    temp = []
    for i in tqdm(range(len(target_final_df))):
        units = ['[', ']', 'tensor', '(', ')']

        for unit in units:
            s = str(target_final_df[target_name][i]).replace(unit, '')
        temp.append(s)

    temp_dict = {target_name: temp}

    final_df = pd.DataFrame(temp_dict)

    return final_df


def tensor_separator(df, column_name):
    to_replace = ["t e n s o r", "[", "]", "(", ")", " ", "n", "/"]
    full_tensor_list =[]
    for tensor in tqdm(df[column_name]):
        # tensor = tensor.astype(str) ## if tensor != str
        tensor = " ".join(tensor)
        list_per_row = []
        for i in to_replace:
            tensor = tensor.lower()
            tensor = tensor.replace(i, "")
        tensor_list = tensor.split(",")
        list_per_row.extend(tensor_list)
        full_tensor_list.append(list_per_row)
    full_tensor_df = pd.DataFrame(full_tensor_list)

    return full_tensor_df

def new_tensor_separator(df, column_name):
    to_replace = ["t e n s o r", "[", "]", "(", ")", " ", "n", "/", "tensor", "=", "device", "d e v i c e", "cuda:0", "c u d a = 0",]
    full_tensor_list =[]
    for tensor in tqdm(df[column_name]):
        # tensor = tensor.astype(str) ## if tensor != str
        tensor = " ".join(tensor)
        list_per_row = []
        for i in to_replace:
            tensor = tensor.lower()
            tensor = tensor.replace(i, "")
        tensor_list = tensor.split(",")
        list_per_row.extend(tensor_list)
        full_tensor_list.append(list_per_row)
    full_tensor_df = pd.DataFrame(full_tensor_list)

    return full_tensor_df

def X2_T2(df1, df2, column):
    X_temp = pd.DataFrame()
    temp_train = df1.drop(columns=column)
    for i in temp_train:
        to_be_X = pd.concat([X_temp, tensor_separator(temp_train, i)], axis=1)

    to_be_X = (pd.concat([to_be_X, df1[column]], axis=1)).astype('float64')

    X_test_temp = pd.DataFrame()
    for i in df2:
        to_be_test_X = (pd.concat([X_test_temp, tensor_separator(df2, i)], axis=1)).astype('float64')

    return to_be_X, to_be_test_X


def test_val_separator(df1, df2, test_size):
    train_cols = df1.columns.values.tolist()
    test_cols = df2.columns.values.tolist()
    column_y = [i for i in train_cols if i not in test_cols]
    column_X = [i for i in train_cols if i not in column_y]
    X = df1[df1.columns[df1.columns.isin(column_X)]]
    y = df1[df1.columns[df1.columns.isin(column_y)]]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    test_X = df2

    return X_train, X_val, y_train, y_val, test_X

class SimpleOps():
    '''
    간단한 pandas 표 자르고 넣기
    매번 치기 귀찮아서 만듦
    '''
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        column = self.df.iloc[:, idx]

    def right_merger(self, df1, column_idx):
        '''
        입력한 두개의 df를 right merge합니다.
        column_idx에는 기준이 되는 컬럼 index를 입력해주면 됩니다.
        '''
        merged = pd.merge(self, df1, how='right', on=self.columns[column_idx])

        return merged

    def left_merger(self, df1, column_idx):
        '''
        right merger 만들었는데 left는 안만들면 섭섭할까봐 만듦
        '''
        merged = pd.merge(self, df1, how='left', on=self.columns[column_idx])
        return merged

    def ccd(self, start_number, end_number):
        '''
        Continuous_Column_Dropper
        연속되는 column을 삭제합니다.

        :param start_number: 시작 column index
        :param end_number: 종료 column index
        :return:
        '''
        df = self.drop(self.columns[start_number:(end_number + 1)], axis=1)
        return df

    def ocd(self, colnum1):
        df = self.drop(columns=[colnum1])
        return df

    def law_train_clean_ccd(self, df):
        df = pd.concat([self.iloc[:, 0], df], axis=1)
        temp = SimpleOps.ccd(self, 3, 4)
        temp = SimpleOps.right_merger(temp, df, 0)
        temptemp = SimpleOps.ccd(self, 1, 3)
        train_cleansed = SimpleOps.right_merger(temp, temptemp, 0)
        return train_cleansed

    def law_train_clean1(self, df, colnum1):
        df = pd.concat([self.iloc[:, 0], df], axis=1)
        temp = SimpleOps.ocd(self, colnum1)
        temp = SimpleOps.right_merger(temp, df, 0)
        temptemp = SimpleOps.ocd(self, colnum1)
        train_cleansed = SimpleOps.right_merger(temp, temptemp, 0)
        return train_cleansed

    def df_divider(self, column):
        if len(self) % 2 == 0:
            divided_df = np.array_split(self[column], 26)

        else:
            divided_df = np.array_split(self[column][:-1], 25)
            divided_df.apend(self[column][-1:])

        return divided_df[0], divided_df[1], divided_df[2], divided_df[3], divided_df[4], divided_df[5], divided_df[6], divided_df[7] , divided_df[8] , divided_df[9], divided_df[10], divided_df[11], divided_df[12], divided_df[13], divided_df[14], divided_df[15], divided_df[16], divided_df[17] , divided_df[18] , divided_df[19], divided_df[20], divided_df[21], divided_df[22], divided_df[23], divided_df[24], divided_df[25]



print(
" ___________________________\n"
"|                           |\n"
"|======== YearDream ========|\n"
"|===========================|\n"
"|==== DLC Well Imported ====|\n"
"|===========================|\n"
"|========= BYJASON =========|\n"
"|________11th_Jun_23________|\n"
)