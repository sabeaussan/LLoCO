import os
import llm_utils
import json

response_format_csv = {
    "type": "json_schema",
    "json_schema": {
        "name": "csv_structure_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "file_name": {"type": "string"},
                "high_level_desc": {"type": "string"},
                "num_rows": {"type": "integer"},
                "num_columns": {"type": "integer"},
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value_type": {"type": "string"},
                            "description": {"type": "string"},
							"index": {"type": "integer"},
                        },
                        "required": ["name", "value_type", "description", "index"],
                        "additionalProperties": False
                    }
                }
            },
            "required": [
                "file_name",
                "high_level_desc",
                "num_rows",
                "num_columns",
                "columns"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}

response_format_scope = {
    "type": "json_schema",
    "json_schema": {
        "name": "problem_refinement",
        "schema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
								"type": "string",
								"description": "The targetd question to the user to refine the problem description or gather missing information."
							},
                            "question_number": {"type": "integer"},
                        },
                        "required": ["question", "question_number"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["questions"],
            "additionalProperties": False
        },
        "strict": True
    }
}


def refine_problem_description(prompt_path, context):
	with open(prompt_path, "r") as f:
		prompt = f.read()
	
	messages = [
		{"role": "system", "content": prompt}, 
		{"role": "user", "content": context},
	]
	# Query the LLM
	response = llm_utils.openai_ask_requests(messages, response_format=response_format_scope)
	return json.loads(response)

def _get_basic_csv_summary(file_path):
	"""
	Get a summary of a CSV file using pandas.
	"""
	import pandas as pd
	import io
	buffer = io.StringIO()
	pd.set_option('display.max_rows', 10)
	pd.set_option('display.max_columns', 10)
	df = pd.read_csv(file_path)
	fname = file_path.split("/")[-1]
	summary = f"*** Input file: {fname}\n"
	df.info(buf=buffer)
	info = buffer.getvalue()
	summary += f"* Global information : \n{info}\n"
	summary += f"* Excerpt from {fname} : \n{df.head().to_json()}\n"
	return summary

def get_csv_files_summary(directory):
	files = os.listdir(directory)
	response = "# INPUT FILES DESCRIPTION\n\n"
	has_csv_file = False
	for file in files:
		if file.endswith(".csv"):
			file_path = os.path.join(directory, file)
			response += _get_basic_csv_summary(file_path)
			has_csv_file = True
	return response, has_csv_file

def convert_file_to_json(directory, complete_description):
	"""
	Convert a CSV file to JSON format.
	"""
	with open("prompts/system_prompt_json.txt", "r") as f:
		p = f.read()
	files = os.listdir(directory)
	response = "# INPUT FILES DESCRIPTION\n\n"
	has_csv_file = False
	for file in files:
		# TODO : add some other file types

		if file.endswith(".csv"):
			file_path = os.path.join(directory, file)
			# Get a basic summary of the CSV file
			csv_summary = _get_basic_csv_summary(file_path)
			messages = [
				{"role": "system", "content": p}, 
				{"role": "user", "content": complete_description},
				{"role": "user", "content": f"# File {file_path} : \n\n{csv_summary}"}
			]
			# Query the LLM
			response += llm_utils.openai_ask_requests(messages, response_format=response_format_csv)
			response += "\n\n"
			has_csv_file = True
	return response, has_csv_file