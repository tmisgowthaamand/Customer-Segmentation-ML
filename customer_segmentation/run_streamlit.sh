#!/bin/bash
cd /app/customer_segmentation
streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0
