import json
from tqdm import tqdm

window_size = 6

with open('densephrases-data/convqa/orquac/original/quac_format/test.txt') as json_file:
    json_data = json.load(json_file)
    dict_original_answers = {}

    for data in json_data['data']:
        for qa in data['paragraphs'][0]['qas']:
            dict_original_answers[qa['id']] = [i['text'] for i in qa['answers']]


with open('densephrases-data/convqa/orquac/original/preprocessed/test.txt') as f:
    total_data = f.readlines()

    dict_final_data = {}
    lst_dict_data = []
    
    for data in total_data:
        dict_data = json.loads(data)

        lst_history_question_answer = []
        
        
        if (dict_data['answer']['text'] == 'CANNOTANSWER') or (dict_data['answer']['text'] == 'NOTRECOVERED'):
            continue

        dict_data['id'] = dict_data.pop('qid')
        dict_data['answers_answerstart'] = dict_data.pop('answer')
        dict_data['answers'] = dict_original_answers[dict_data['id']]#[dict_data['answers_answerstart']['text']]
        dict_data['current_question'] = dict_data['question']

        for h in dict_data['history']:
            if (h['answer']['text'] != 'CANNOTANSWER') and (h['answer']['text'] != 'NOTRECOVERED'):
                lst_history_question_answer.append(h['answer']['text'])
                lst_history_question_answer.append(h['question'])

        
        # to obtain the recent history first
        lst_history_question_answer_reverse = lst_history_question_answer.copy()

        lst_history_question_answer_reverse.reverse()
        lst_history_question_answer_reverse = lst_history_question_answer_reverse[:window_size]

        lst_history_question_answer_reverse_current_question = [dict_data['question']] + lst_history_question_answer_reverse

        question_text = ' [SEP] '.join(lst_history_question_answer_reverse_current_question) 

        #import pdb; pdb.set_trace()        

        del dict_data['question']
        dict_data['question'] = question_text

        del dict_data['history']
        del dict_data['evidences']
        del dict_data['retrieval_labels']
        del dict_data['answers_answerstart']
        
        #import pdb; pdb.set_trace()
        
        lst_dict_data.append(dict_data)

        
        
    dict_final_data['data'] = lst_dict_data

        #import pdb; pdb.set_trace()

with open('densephrases-data/convqa/orquac/preprocessed/qa/test_answer_'+str(window_size)+'.json', "w") as writer: 
    writer.write(json.dumps(dict_final_data, indent=4) + "\n")    

