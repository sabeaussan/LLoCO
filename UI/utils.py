import sys
import threading
import time
import random 


def show_logo():
	# TODO : add check working directory before loading the logo
	with open("UI/logo.txt", "r") as file:
		logo = file.read()
	print(logo)
      
class Spinner(threading.Thread):
	def __init__(self, description="Doing some work ...  "):
		super().__init__()
		self.spinner_active = False
		self.description = description

	def run(self):
		# ANSI escape codes for colors
		colors = [
			'\033[91m',  # Red
			'\033[92m',  # Green
			'\033[93m',  # Yellow
			'\033[94m',  # Blue
			'\033[95m',  # Purple
			'\033[96m',  # Cyan
		]
		# Reset color to default
		reset_color = '\033[0m'
		robot = "🤖 "
		spin_chars = "⠁,⠃,⠇,⠧,⠷,⠿,⠻,⠽,⠯,⠟".split(',')
		offset = 15
		while True:
			for i in range(len(spin_chars)):
				color = random.choice(colors)
				chars = " ".join([spin_chars[(i+j+2)%len(spin_chars)] for j in range(offset)])
				sys.stdout.write('\r'+ robot + self.description+ color + chars + reset_color)
				sys.stdout.flush()
				time.sleep(0.03)

			if not self.spinner_active:
				# Clear the previous spinner line completely
				line_length = len(robot + self.description + " " + chars)
				sys.stdout.write('\r' + ' ' * line_length)  # overwrite with spaces and return
				sys.stdout.flush()
				sys.stdout.write('\r' + robot + self.description + " Done !\n")  # print final message
				sys.stdout.flush()
				break


class SpinnerManager(object):

	def __init__(self, text_desc, active=True):
		self.text_desc = text_desc
		self.active = active

	def __enter__(self):
		if self.active:
			self.spinner = Spinner(self.text_desc)
			self.spinner.start()
			self.spinner.spinner_active = True

	def __exit__(self, type, value, traceback):
		if self.active:
			self.spinner.spinner_active = False
			self.spinner.join()