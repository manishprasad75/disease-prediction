import json


def is_json(data):
    try:
        p_data = json.loads(data)
        valid = True
    except ValueError:
        valid = False
    return valid


def convert_into_array(data):
    try:
        data = data.split(",")
        actual_data = []
        for x in data:
            actual_data.append(x.strip())
        return actual_data
    except:
        return []