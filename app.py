# -*- coding:utf8 -*-
# !/usr/bin/env python3
# Copyright 2025 Volodymyr Pavliukevych
# Author Volodymyr Pavliukevych

"""This is a streamlit app main file.
"""

from typing import Any
import streamlit as st
from src.poc_page import WelcomePage, TemplatePage

class Kernel():
	def __init__(self) -> None:
		self.model_cache = {}

	def set_model(self, model, key: str) -> None:
		self.model_cache[key] = model

	def get_model(self, key: str) -> Any:
		return self.model_cache[key]

	def serve(self) -> None:
		"""Proof of Concepts and Beyond """
		st.set_page_config(
			page_title="Proof of Concepts and Beyond",
			layout="wide",
			page_icon="images/favicon.ico"
			)
		st.title("Proof of Concepts and Beyond")
		
		menu = ["Template matching", "Home"]
		choice = st.sidebar.selectbox("Concepts available to view", menu)

		if choice == "Home":
			WelcomePage(kernel=self).draw()

		elif choice == "Template matching":
			TemplatePage(kernel=self).draw()
		
		st.sidebar.markdown('''<div id="container" style="height: 100%;border-collapse: collapse; padding: 0; margin: 0;"> <div id="copyright" style="padding: 0; margin: 0;"> <small>Volodymyr Pavliukevych, 2025</small></div></div>''', unsafe_allow_html=True)


def main():
	Kernel().serve()
if __name__ == '__main__':
	main()