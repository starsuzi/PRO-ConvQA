import json
from tqdm import tqdm

window_size = 6

with open('densephrases-data/convqa/topiocqa/original/original/topiocqa_dev.json') as json_file, open('densephrases-data/convqa/topiocqa/original/all_history/dev.json') as all_history_json_file:
    json_data = json.load(json_file)
    all_history_json_data = json.load(all_history_json_file)

    dict_final_data = {}
    lst_dict_data = []
    lst_history_question_answer = []

    # TODO
    lst_question = []
    idx = 0

    for dict_data in zip(json_data, all_history_json_data):
        assert dict_data[0]['Conversation_no'] == dict_data[1]['conv_id']
        assert dict_data[0]['Turn_no'] == dict_data[1]['turn_id']

        dict_title_para = {}
        dict_paragraphs = {}
        dict_qas = {}
        #lst_history_question_answer = []

        if (dict_data[0]['Rationale'] == '') or (dict_data[1]['answers'][0] == 'UNANSWERABLE'):
            #import pdb; pdb.set_trace()
            continue

        if dict_data[1]['answers'][0] in dict_data[0]['Rationale']:
            answer_text = dict_data[1]['answers'][0]
        else:
            answer_text = dict_data[0]['Rationale']
            if answer_text[0] == " ":
                answer_text = answer_text[1:]
            if answer_text[-1] == " ":
                answer_text = answer_text[:-1]
            if answer_text[0] == "\n":
                answer_text = answer_text[1:]
            if answer_text[-1] == "\n":
                answer_text = answer_text[:-1]

        if answer_text not in dict_data[1]['positive_ctxs'][0]['text']:
            #import pdb; pdb.set_trace()
            continue

        lst_additional_answer = []
        for additional_answer in dict_data[0]['Additional_answers']:
            if additional_answer['Rationale'] == '':
                continue
            else:
                if additional_answer['Answer'] in additional_answer['Rationale']:
                    additional_answer_text = additional_answer['Answer']
                else:
                    additional_answer_text = additional_answer['Rationale']
                    if additional_answer_text[0] == " ":
                        additional_answer_text = additional_answer_text[1:]
                    if additional_answer_text[-1] == " ":
                        additional_answer_text = additional_answer_text[:-1]
                    if additional_answer_text[0] == "\n":
                        additional_answer_text = additional_answer_text[1:]
                    if additional_answer_text[-1] == "\n":
                        additional_answer_text = additional_answer_text[:-1]

                if additional_answer_text not in dict_data[1]['positive_ctxs'][0]['text']:
                    continue

                additional_answer_start = dict_data[1]['positive_ctxs'][0]['text'].index(additional_answer_text)
                lst_additional_answer.append({"text": additional_answer_text, "answer_start": additional_answer_start})
                #import pdb; pdb.set_trace()
                

        dict_title_para['title'] = dict_data[1]['positive_ctxs'][0]['title']
        dict_paragraphs['context'] = dict_data[1]['positive_ctxs'][0]['text']
        current_conv_id = dict_data[0]['Conversation_no']
        current_turn_id = dict_data[0]['Turn_no']

        if lst_dict_data != []:
            previous_conv_id = lst_dict_data[-1]['paragraphs'][0]['qas'][0]['conv_id']

            if previous_conv_id != current_conv_id:
                lst_history_question_answer = []
                lst_history_question_answer.append(answer_text)
                lst_history_question_answer.append(dict_data[0]['Question'])
        
            else:
                lst_history_question_answer.append(answer_text)
                lst_history_question_answer.append(dict_data[0]['Question'])

        else: 
            previous_conv_id = -1
            lst_history_question_answer.append(answer_text)
            lst_history_question_answer.append(dict_data[0]['Question'])     

        
        answer_start = dict_data[1]['positive_ctxs'][0]['text'].index(answer_text)

        # to obtain the recent history first
        lst_history_question_answer_reverse = lst_history_question_answer[:-2]

        lst_history_question_answer_reverse.reverse()
        lst_history_question_answer_reverse = lst_history_question_answer_reverse[:window_size]

        lst_history_question_answer_reverse_current_question = [dict_data[0]['Question']] + lst_history_question_answer_reverse

        question_text = ' [SEP] '.join(lst_history_question_answer_reverse_current_question) 

        # TODO
        lst_question.append(question_text)


        if (idx - 1) > 0:
            previous_question_text = lst_question[idx - 1]

        if (previous_conv_id != current_conv_id) or (idx == 0):
            #import pdb; pdb.set_trace()
            previous_question_text = question_text

        idx = idx + 1

        dict_qas['previous_question'] = previous_question_text


        dict_qas['question'] = question_text

        dict_qas['orig_answers'] = [{'text':answer_text, 'answer_start':answer_start}]
        dict_qas['answers'] = lst_additional_answer + dict_qas['orig_answers']

        qid = 'dev_' + str(dict_data[0]['Conversation_no']) + '_' + str(dict_data[0]['Turn_no'])
        dict_qas['id'] = qid
        dict_qas['conv_id'] = dict_data[0]['Conversation_no']
        dict_qas['turn_id'] = dict_data[0]['Turn_no']


        dict_paragraphs['qas'] = [dict_qas]
        dict_title_para['paragraphs'] = [dict_paragraphs]

        lst_dict_data.append(dict_title_para)
        #import pdb; pdb.set_trace()

        

    dict_final_data['data'] = lst_dict_data



with open('densephrases-data/convqa/topiocqa/preprocessed/retriever/dev_answer_'+str(window_size)+'_hisContra.json', "w") as writer: 
    writer.write(json.dumps(dict_final_data, indent=4) + "\n")  