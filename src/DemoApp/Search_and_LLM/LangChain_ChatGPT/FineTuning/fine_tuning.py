import csv
import json

def csv_to_jsonl(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        with open(output_file, mode='w', encoding='utf-8') as jsonlfile:
            for row in reader:
                message = []
                
                message.append({
                    "role": "system",
                    "content": "A secretary who is close to the user and anticipates potential needs" # Agents for predicting user needs"
                })
                
                message.append({
                    "role": "user",
                    "content": row['user']
                })
                
                message.append({
                    "role": "assistant",
                    "content": row['assistant']
                })

                data = {"messages": message}
                jsonlfile.write(json.dumps(data, ensure_ascii=False) + '\n')


# csv_to_jsonl('test_data.csv', 'test_data.jsonl')

# csv_to_jsonl('user_data.csv', 'user_data.jsonl')
# csv_to_jsonl('user_data_NotRecommend.csv', 'user_data_NotRecommend.jsonl')

# generate prompt
# csv_to_jsonl('user_data_details_2.csv', 'user_data_details.jsonl')
csv_to_jsonl('user_data_details_3.csv', 'user_data_details_3.jsonl')
