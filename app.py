import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from datetime import datetime
import db
#import utils.db as db


COMMENT_TEMPLATE_MD = """{} - {}
> {}"""
st.title("Team Anything Is Fine")
st.sidebar.title("Select Your Choices")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("## Product Review Analysis")
#st.sidebar.markdown("A making of sentiment analysis on different businesses products ")

data_path = ("Yelp.csv")

def load_data():
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    return data

data = load_data()

st.markdown("")
see_data = st.expander('Click here to see the dataset')
with see_data:
        st.dataframe(data.reset_index(drop=True))
st.text('')

#Line Chart
#st.line_chart(data['sentiments'])
#st.dataframe(data)

st.sidebar.subheader("Show Random Reviews")
random_tweet = st.sidebar.radio('Select the Sentiment',('positive','negative','neutral'))
if st.sidebar.checkbox("Show", False, key="1"):
    st.subheader("Here are some of random reviews according to your choice!")
    for i in range(len(data['date'])):
        if i ==5:
            break
        else:
            st.markdown(str(i+1) +"." + data.query("sentiments == @random_tweet")[['text']].sample(n=1).iat[0,0])

st.sidebar.markdown("### Visualization of Reviews")
select = st.sidebar.selectbox('Select type of visualization',['Histogram','PieChart'])

sentiment_count = data['sentiments'].value_counts()
sentiment_count = pd.DataFrame({'Sentiments':sentiment_count.index,'Reviews':sentiment_count.values})

if st.sidebar.checkbox('Show',False,key='0'):
    st.markdown("### No. of reviews by sentiments ")
    if select=='Histogram':
        fig = px.bar(sentiment_count,x='Sentiments',y='Reviews',color='Reviews',height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count,values='Reviews',names='Sentiments')
        st.plotly_chart(fig)

st.sidebar.subheader("Breakdown Sentiments by city")
choice = st.sidebar.multiselect("Pick City", tuple(pd.unique(data["city"])))
if st.sidebar.checkbox("Show", False, key="5"):
    if len(choice) > 0:
        chosen_data = data[data["city"].isin(choice)]
        fig = px.histogram(chosen_data, x="city", y="sentiments",
                                histfunc="count", color="sentiments",
                                facet_col="sentiments", labels={"sentiments": "sentiment"})
        st.plotly_chart(fig)

# Word cloud
st.sidebar.subheader("Word Cloud")
word_sentiment = st.sidebar.radio("Which Sentiment to Display?", tuple(pd.unique(data["sentiments"])))
if st.sidebar.checkbox("Show", False, key="6"):
    st.subheader(f"Word Cloud for {word_sentiment.capitalize()} Sentiment")
    df = data[data["sentiments"]==word_sentiment]
    words = " ".join(df["text"])
    #processed_words = " ".join([word for word in words.split() if "http" not in word and not word.startswith() and word != "RT"])
    processed_words = " ".join([word for word in words.split()])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=600, height=500).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()


# Reviews part
conn = db.connect()
comments = db.collect(conn)

with st.expander("💬 Review our project"):

    # Show comments
    #st.write("**Comments:**")

    for index, entry in enumerate(comments.itertuples()):
        st.markdown(COMMENT_TEMPLATE_MD.format(entry.name, entry.date, entry.comment))
        is_last = index == len(comments) - 1
        is_new = "just_posted" in st.session_state and is_last
        if is_new:
            st.success("☝️ Your comment was successfully posted.")

    # Insert comment
    #st.write("**Add your own comment:**")
    form = st.form("comment",clear_on_submit=True)
    name = form.text_input("Name")
    comment = form.text_area("Comment")
    submit = form.form_submit_button("Add comment")

    if submit:
        date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        db.insert(conn, [[name, comment, date]])
        if "just_posted" not in st.session_state:
            st.session_state["just_posted"] = True
        st.experimental_rerun()