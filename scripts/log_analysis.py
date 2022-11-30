import os, subprocess
import time
import openpyxl as op
import argparse
import re

#Analyze haystack log script

parser = argparse.ArgumentParser(description='argparse', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(type=str, default='log.log', dest='logfile', help='Log file to be analyzed')
args = parser.parse_args()
logfile = args.logfile

def get_pipeline_data(logfile):
    cmd1 = "cat %s |grep Query |grep preprocess |awk '{print $1, $11, $19}'" % (logfile)
    cmd2 = "cat %s |grep Query |grep interval |awk '{print $1, $11, $17}'" % (logfile)
    cmd3 = "cat %s |grep Query |grep postprocess |awk '{print $1, $11, $17}'" % (logfile)
    cmd4 = "cat %s |grep Retriever |grep preprocess |awk '{print $1, $11, $19}'" % (logfile)
    cmd5 = "cat %s |grep Retriever |grep interval |awk '{print $1, $11, $17}'" % (logfile)
    cmd6 = "cat %s |grep Retriever |grep postprocess |awk '{print $1, $11, $17}'" % (logfile)
    cmd7 = "cat %s |grep Ranker |grep preprocess |awk '{print $1, $11, $19}'" % (logfile)
    cmd8 = "cat %s |grep query_encode_time |awk '{print $1, $11, $13}'" % (logfile)
    cmd9 = "cat %s |grep docs_encode_time |awk '{print $1, $11, $13}'" % (logfile)
    cmd10 = "cat %s |grep score_time |awk '{print $1, $11, $13}'" % (logfile)
    cmd11 = "cat %s |grep Ranker |grep interval |awk '{print $1, $11, $17}'" % (logfile)
    cmd12 = "cat %s |grep Ranker |grep postprocess |awk '{print $1, $11, $17}'" % (logfile)
    cmd13 = "cat %s |grep Docs2Answers |grep preprocess |awk '{print $1, $11, $19}'" % (logfile)
    cmd14 = "cat %s |grep Docs2Answers |grep interval |awk '{print $1, $11, $17}'" % (logfile)
    cmd15 = "cat %s |grep Docs2Answers |grep postprocess |awk '{print $1, $11, $17}'" % (logfile)
    cmd16 = "cat %s |grep 'end2end time' |awk -F ' |}' '{print $1, $14, $19}'" % (logfile)

    query_pre_time = subprocess.getoutput(cmd1).split('\n')
    query_time = subprocess.getoutput(cmd2).split('\n')
    query_post_time = subprocess.getoutput(cmd3).split('\n')
    retriever_pre_time = subprocess.getoutput(cmd4).split('\n')
    retriever_time = subprocess.getoutput(cmd5).split('\n')
    retriever_post_time = subprocess.getoutput(cmd6).split('\n')
    ranker_pre_time = subprocess.getoutput(cmd7).split('\n')
    query_encode_time = subprocess.getoutput(cmd8).split('\n')
    docs_encode_time = subprocess.getoutput(cmd9).split('\n')
    score_time = subprocess.getoutput(cmd10).split('\n')
    ranker_time = subprocess.getoutput(cmd11).split('\n')
    ranker_post_time = subprocess.getoutput(cmd12).split('\n')
    Docs2Answers_pre_time = subprocess.getoutput(cmd13).split('\n')
    Docs2Answers_time = subprocess.getoutput(cmd14).split('\n')
    Docs2Answers_post_time = subprocess.getoutput(cmd15).split('\n')
    e2e_time = subprocess.getoutput(cmd16).split('\n')
    # Faiss or Embedding retriever pipeline
    cmd_17 = "cat %s |grep embed_query_time |awk '{print $1, $11, $13}'" % (logfile)
    cmd_18 = "cat %s |grep query_by_embedding_time |awk '{print $1, $11, $13}'" % (logfile)
    embed_query_time = subprocess.getoutput(cmd_17).split('\n')
    query_by_embedding_time = subprocess.getoutput(cmd_18).split('\n')

    if len(embed_query_time[0]) == 0:
        print('colbert pipeline')
        data = [query_pre_time, query_time, query_post_time, retriever_pre_time, retriever_time, retriever_post_time, ranker_pre_time, query_encode_time, docs_encode_time, score_time, ranker_time, ranker_post_time, Docs2Answers_pre_time, Docs2Answers_time, Docs2Answers_post_time, e2e_time]
    else:
        print('faiss or Embedding retriever pipeline')
        data = [query_pre_time, query_time, query_post_time, retriever_pre_time, embed_query_time, query_by_embedding_time, retriever_time, retriever_post_time, Docs2Answers_pre_time, Docs2Answers_time, Docs2Answers_post_time, e2e_time]

    for i in range(len(data)):
        list_data = []
        for j in range(len(data[i])):
            num = int(data[i][j].split()[1].split('_')[1].replace('\'',''))
            thread_id = int(data[i][j].split()[1].split('_')[0].replace('\'',''))
            list1 = data[i][j].split()
            list1.append(num)
            list1.append(thread_id)
            list_data.append(list1)
        list_data.sort(key=lambda list_data:list_data[-1])
        list_data.sort(key=lambda list_data:list_data[-2])
        data[i] = list_data
    return data

def op_toexcel(data, filename):
    wb = op.Workbook()
    ws = wb['Sheet']
    if len(data) == 16:
        ws.append(['query_pre_time','query_time','query_post_time', 'retriever_pre_time', 'retriever_time', 'retriever_post_time', 'ranker_pre_time', 'query_encode_time', 'docs_encode_time', 'score_time', 'ranker_time', 'ranker_post_time', 'Docs2Answers_pre_time', 'Docs2Answers_time', 'Docs2Answers_post_time', 'e2e_time', 'request_id', 'container'])
        for i in range(len(data[0])):
            data[0][i][0] = re.search('haystack-api.*', data[0][i][0]).group()
            row = data[0][i][2], data[1][i][2], data[2][i][2], data[3][i][2], data[4][i][2], data[5][i][2], data[6][i][2], data[7][i][2], data[8][i][2], data[9][i][2], data[10][i][2], data[11][i][2], data[12][i][2], data[13][i][2], data[14][i][2], data[15][i][2], data[0][i][1], data[0][i][0]
            ws.append(row)
            time.sleep(0.1)
        wb.save(filename)
    else:
        ws.append(['query_pre_time', 'query_time', 'query_post_time', 'retriever_pre_time', 'embed_query_time', 'query_by_embedding_time', 'retriever_time', 'retriever_post_time', 'Docs2Answers_pre_time', 'Docs2Answers_time', 'Docs2Answers_post_time', 'e2e_time', 'request_id', 'container'])
        for i in range(len(data[0])):
            data[0][i][0] = re.search('haystack-api.*', data[0][i][0]).group()
            row = data[0][i][2], data[1][i][2], data[2][i][2], data[3][i][2], data[4][i][2], data[5][i][2], data[6][i][2], data[7][i][2], data[8][i][2], data[9][i][2], data[10][i][2], data[11][i][2], data[0][i][1], data[0][i][0]
            ws.append(row)
            time.sleep(0.1)
        wb.save(filename)
    print("Log analysis completed, the resulting file:", filename)

filename = 'output.xlsx'
testData = get_pipeline_data(logfile)
op_toexcel(testData,filename)
