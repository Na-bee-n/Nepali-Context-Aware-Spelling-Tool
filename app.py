import streamlit as st
import paragraph  # Import paragraph.py
import sentence  # Import sentence.py


def main():

    mode = st.sidebar.selectbox("Select Mode", ("Sentence", "Paragraph"))

    if mode == "Sentence":
        st.write("### Sentence Mode")
        sentence.main()  # Call the sentence mode function

    elif mode == "Paragraph":
        st.write("### Paragraph Mode")
        paragraph.main()  # Call the paragraph mode function


if __name__ == "__main__":
    main()
