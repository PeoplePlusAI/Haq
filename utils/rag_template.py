import requests
import json
import redis

def rag_(data):
    url = 'localhost:8000/api/chat'
    headers = {'Content-Type: application/json'}
    data.update({
        'messages': [
            {
                'role': 'user',
                'content': 'Hello'
            }
        ]
    })
    # curl --location 'localhost:8000/api/chat' --header 'Content-Type: application/json' --data '{ "messages": [{ "role": "user", "content": "Hello" }] }'
    response = requests.post(url, headers=headers, data=data)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Process the response
        return response
    else:
        return None


def input_message(chat_id, message):
    history = get_redis(chat_id)
    history = json.loads(history)
    thread_id = history["thread_id"]
    run_id = history["run_id"]
    status = history["status"]
    thread = client.beta.threads.fetch(thread_id)
    run = client.beta.threads.runs.fetch(thread_id, run_id)
    tool_output_array = []
    tool = client.beta.threads.tools.fetch(thread_id, run.tool_call_id)
    if status == "success":
        tool_output_array.append(
            {
                "tool_call_id": tool.id,
                "output": message
            }
        )
        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_output_array
        )
        run, status = get_run_status(run, client, thread)

        message = get_assistant_message(client, thread.id)

        history = {
            "thread_id": thread.id,
            "run_id": run.id,
            "status": status,
        }
        history = json.dumps(history)
        set_redis(chat_id, history)
        return message, history
    else:
        return "Complaint failed", history

def rag_(data):
    url = 'localhost:8000/api/chat'
    headers = {'Content-Type: application/json'}
    data.update({
        'messages': [
            {
                'role': 'user',
                'content': 'Hello'
            }
        ]
    })
    # curl --location 'localhost:8000/api/chat' --header 'Content-Type: application/json' --data '{ "messages": [{ "role": "user", "content": "Hello" }] }'
    response = requests.post(url, headers=headers, data=data)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Process the response
        return response
    else:
        return None
    

# def file_complaint(data):
#     headers = {'Content-Type': 'application/json'}
#     data = {
#     "service": {
#         "tenantId": "pg.cityb",
#         "serviceCode": "NoStreetlight",
#         "description": "",
#         "additionalDetail": {},
#         "source": "web",
#         "address": {
#             "city": data["city"],
#             "district": data["district"],
#             "region": data["region"],
#             "state": data["state"],
#             "locality": {
#                 "code": "SUN11",
#                 "name": data["locality"]
#             },
#             "geoLocation": {}
#         }
#     },
#     "workflow": {
#         "action": "APPLY"
#     },
#     "RequestInfo": {
#         "apiId": "Rainmaker",
#         "authToken": data["auth_token"],
#         "userInfo": {
#             "id": 2079,
#             "uuid": "7e2b023a-2f7f-444c-a48e-78d75911387a",
#             "userName": data["username"],
#             "name": data["name"],
#             "mobileNumber": data["username"],
#             "emailId": "",
#             "locale": None,
#             "type": "CITIZEN",
#             "roles": [
#                 {
#                     "name": "Citizen",
#                     "code": "CITIZEN",
#                     "tenantId": "pg"
#                 }
#             ],
#             "active": True,
#             "tenantId": "pg",
#             "permanentCity": "pg.citya"
#         },
#         "msgId": "1703653602370|en_IN",
#         "plainAccessRequest": {}
#     }
# }
#     url = "https://staging.digit.org/pgr-services/v2/request/_create"

#     response = requests.post(url, headers=headers, data=json.dumps(data))

#     if response.status_code == 200:
#         response_data = response.json()
#         return response_data
#     else:
#         print(response.content)
#         print(f"Error: {response.status_code}")
#         return None
    
# def search_complaint(data):
#     headers = {'Content-Type': 'application/json'}
#     mobile_number = data["username"]
#     url = f"https://staging.digit.org/pgr-services/v2/request/_search?tenantId=pg.cityb&mobileNumber={mobile_number}&_=1704443852959"

#     data = {
#         "RequestInfo":{
#             "apiId":"Rainmaker",
#             "authToken":data["auth_token"],
#             "userInfo":{
#                 "id":2079,
#                 "uuid":"7e2b023a-2f7f-444c-a48e-78d75911387a",
#                 "userName":data["username"],
#                 "name":data["name"],
#                 "mobileNumber":data["username"],
#                 "emailId":"",
#                 "locale":None,
#                 "type":"CITIZEN",
#                 "roles":[
#                     {
#                         "name":"Citizen",
#                         "code":"CITIZEN",
#                         "tenantId":"pg"
#                     }
#                 ],
#                 "active":True,
#                 "tenantId":"pg",
#                 "permanentCity":"pg.cityb"
#             },
#             "msgId":"1704443852959|en_IN",
#             "plainAccessRequest":{}
#         }
#     }

#     response = requests.post(url, headers=headers, data=json.dumps(data))

#     if response.status_code == 200:
#         response_data = response.json()
#         return response_data
#     else:
#         print(response.content)
#         print(f"Error: {response.status_code}")
#         return None

    

