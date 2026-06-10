from transformers import pipeline
import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup

st.title("レビュー分析アプリ")

url = st.text_input("価格.comレビューURL")

#レビュー件数スライダー
review_limit = st.slider(
    "分析するレビュー件数",
    min_value=5,
    max_value=50,
    value=20
)

# モデル読み込み
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

if url:

    #st.write("ページ取得中...")

    #レビュー全件
    all_reviews = []

    try:
        # response = requests.get(
        #     url,
        #     timeout=10
        # )
        # response.raise_for_status()

        for page in range(1, 100):

            if page == 1:
                page_url = url
            else:
                page_url = url.replace(
                    "#tab",
                    f"?Page={page}#tab"
                )

            response = requests.get(
                page_url,
                timeout=10
            )

            
            if response.status_code != 200:
                break

            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(
                    response.text,
                    "html.parser"
                )
            

            page_reviews = soup.find_all(
                "p",
                class_="revEntryCont"
            )

            #レビュー数＝0は終了
            if len(page_reviews) == 0:
                break


            all_reviews.extend(page_reviews)


    except Exception as e:
        st.error(
            f"ページ取得エラー: {e}"
        )
        st.stop()





    #if len(reviews) == 0:
    if len(all_reviews) == 0:        
        st.warning(
            "レビューが見つかりませんでした"
        )
        st.stop()

    #スライダー分の分析
    all_reviews = all_reviews[:review_limit]

    classifier = load_model()



    #st.write("モデル読み込み中...")
    #st.write("レビュー取得中...")




    review_list = []
    sentiment_list = []

    #for review in reviews:
    for review in all_reviews:

        text = review.get_text(strip=True)

        # 空レビュー対策
        if text == "":
            continue

        try:
            result = classifier(text[:512])

            sentiment = result[0]["label"]

            label_map = {
                "positive": "ポジティブ",
                "negative": "ネガティブ",
                "neutral": "中立"
            }

            sentiment = label_map.get(
                sentiment.lower(),
                sentiment
            )

            review_list.append(text)
            sentiment_list.append(sentiment)

        except Exception as e:
            st.error(f"エラー発生: {e}")

    #DF作成
    df = pd.DataFrame({
        "review": review_list,
        "sentiment": sentiment_list
    })

    #ネガティブ抽出
    negative_df = df[
        df["sentiment"] == "ネガティブ"
    ]

    #感情分析失敗時の表示
    if len(df) == 0:
        st.warning(
            "分析できるレビューがありませんでした"
        )
        st.stop()

    #感情分析の集計を記載
    sentiment_counts = df["sentiment"].value_counts()

    st.subheader("感情分析結果")


    #スコア表示
    total = len(df)

    positive_rate = (
        sentiment_counts.get("ポジティブ", 0)
        / total
        * 100
    )
    st.metric(
        "ポジティブ率",
        f"{positive_rate:.1f}%"
    )

    st.write(f"😊 ポジティブ: {sentiment_counts.get('ポジティブ', 0)}件")
    st.write(f"😐 中立: {sentiment_counts.get('中立', 0)}件")
    st.write(f"😞 ネガティブ: {sentiment_counts.get('ネガティブ', 0)}件")



    #集計結果表
    #st.write(sentiment_counts)

    #レビュー取得件数
    st.info(f"レビュー取得件数: {len(review_list)}件")



    #棒グラフ
    #st.write(sentiment_counts)
    st.bar_chart(sentiment_counts)

    #分析結果
    st.dataframe(df)
    
    #ネガティブ抽出の表示
    st.subheader("ネガティブレビュー一覧")

    st.info(
    f"ネガティブレビュー件数: {len(negative_df)}件")

    if len(negative_df) > 0:
        st.dataframe(negative_df)
    else:
        st.success(
            "ネガティブレビューはありませんでした"
        )
    
    
    
    #CSV出力
    csv = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="CSVダウンロード",
        data=csv,
        file_name="review_analysis.csv",
        mime="text/csv"
    ) 