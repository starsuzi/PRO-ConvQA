import json
from tqdm import tqdm

window_size = 6

with open('densephrases-data/convqa/orquac/original/preprocessed/train.txt') as f:
    total_data = f.readlines()

    dict_final_data = {}
    lst_dict_data = []

    # TODO
    lst_question = []
    idx = 0
    
    for data in total_data:
        dict_data = json.loads(data)

        dict_title_para = {}
        dict_paragraphs = {}
        dict_qas = {}
        
        lst_history_question_answer = []
        
        if (dict_data['answer']['text'] == 'CANNOTANSWER') or (dict_data['answer']['text'] == 'NOTRECOVERED'):
            continue

        dict_title_para['title'] = ''
        
        evidence_index = dict_data['retrieval_labels'].index(1)
        dict_paragraphs['context'] = dict_data['evidences'][evidence_index]

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

        # TODO
        lst_question.append(question_text)

        current_turn = int(dict_data['qid'].split('#')[1])
        #import pdb; pdb.set_trace()

        if (idx - 1) > 0:
            previous_question_text = lst_question[idx - 1]

        if (current_turn == 0) or (idx == 0):
            #import pdb; pdb.set_trace()
            previous_question_text = question_text

        idx = idx + 1

        dict_qas['previous_question'] = previous_question_text


        dict_qas['question'] = question_text
        dict_qas['answers'] = [dict_data['answer']]
        dict_qas['id'] = dict_data['qid']
        
        dict_paragraphs['qas'] = [dict_qas]

        dict_title_para['paragraphs'] = [dict_paragraphs]

        

        lst_dict_data.append(dict_title_para)
        

    dict_final_data['data'] = lst_dict_data


with open('densephrases-data/convqa/orquac/preprocessed/retriever/train_answer_'+str(window_size)+'_hisContra.json', "w") as writer: 
    writer.write(json.dumps(dict_final_data, indent=4) + "\n")    

