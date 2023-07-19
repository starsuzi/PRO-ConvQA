import json

with open('densephrases-data/convqa/topiocqa/preprocessed/retriever/train_answer_6_hisContra.json') as json_file:
    json_data = json.load(json_file)

    dict_final_data = {}
    lst_dict_data = []

    for data in json_data['data']:
        dict_data = {}

        dict_data['id'] = data['paragraphs'][0]['qas'][0]['id']
        dict_data['question'] = data['paragraphs'][0]['qas'][0]['question']
        dict_data['answers'] = [data['paragraphs'][0]['qas'][0]['answers'][0]['text']]

        lst_dict_data.append(dict_data)

        #import pdb; pdb.set_trace()
    
    dict_final_data['data'] = lst_dict_data

with open('densephrases-data/convqa/topiocqa/preprocessed/qa/train_answer_6.json', "w") as writer: 
    writer.write(json.dumps(dict_final_data, indent=4) + "\n") 