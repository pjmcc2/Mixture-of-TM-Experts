import streamlit as st
import pandas as pd
import numpy as np
import os

FILE_DIR = "D:/pjmcc/Documents/OSU_datasets/VocalSet/Example"
SAMPLE_FILE_PATH = "D:/pjmcc/Documents/OSU_datasets/VocalSet/Example/f1_arpeggios_belt_c_u.wav"

audio_file = open(SAMPLE_FILE_PATH,'rb')
audio_data1 = audio_file.read()
audio_data2 = open("D:/pjmcc/Documents/OSU_datasets/VocalSet/Example/f1_arpeggios_belt_c_a.wav", 'rb').read()
audio_data3 = open("D:/pjmcc/Documents/OSU_datasets/VocalSet/Example/f1_arpeggios_belt_c_i.wav", 'rb').read()

rng = np.random.default_rng()
fake_data = rng.uniform(size=50) /2 + 0.25
fake_df = pd.DataFrame({"Days": np.arange(50), "Effect": fake_data})

st.header("Sung Vowel Classification")
col1, col2, col3 = st.columns(3, gap="large")
with st.sidebar:
    with st.container():
        st.line_chart(fake_df, x="Days", y="Effect")
        st.table(fake_df.head())
    with col1:
        st.text('Belted C Major arpeggio on \'u\' vowel.')
        st.audio(audio_data1)
    with col2:
        st.text('Belted C Major arpeggio on \'a\' vowel.')
        st.audio(audio_data2)
    with col3:
        st.text('Belted C Major arpeggio on \'i\' vowel.')
        st.audio(audio_data3)


    #with st.button()
