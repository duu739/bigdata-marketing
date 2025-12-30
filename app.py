import streamlit as st
import pandas as pd
import numpy as np
import cv2
import requests
import plotly.express as px
import plotly.graph_objects as go
from googleapiclient.discovery import build
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from openai import OpenAI

# ==========================================
# 1. API 金鑰與初始化
# ==========================================
OPENAI_API_KEY = "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"
YOUTUBE_API_KEY = "AIzaSyDTWvLm7NJ24_4PdY7uK3JDAsodISYbIx0"
client = OpenAI(api_key=OPENAI_API_KEY)

def get_youtube_service():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# ==========================================
# 2. 影像辨識模組
# ==========================================
def analyze_advanced_vision(img_array):
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None: return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    亮度 = np.mean(gray)
    對比度 = np.std(gray)
    飽和度 = np.mean(hsv[:, :, 1])

    face_cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    有臉 = 0
    if not face_cascade.empty():
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        有臉 = 1 if len(faces) > 0 else 0

    edges = cv2.Canny(gray, 100, 200)
    複雜度 = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1]) * 100

    return [亮度, 對比度, 飽和度, 複雜度, 有臉]

def analyze_title_features(title):
    長度 = len(title)
    含數字 = 1 if any(char.isdigit() for char in title) else 0
    含標點 = 1 if any(p in title for p in ['?', '!', '？', '！']) else 0
    return [長度, 含數字, 含標點]

# ==========================================
# 3. K-means 自動分群
# ==========================================
def calculate_clustering_logic(df):
    feature_cols = ['亮度','對比度','飽和度','複雜度','有臉']
    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    wcss = []
    max_k = min(len(df), 10)
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # Elbow Method 選最佳 k
    diff = np.diff(wcss)
    diff_ratio = diff[1:] / diff[:-1]
    optimal_k = np.argmax(diff_ratio < 0.5) + 2  # 自動選群數
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    df['風格'] = kmeans.fit_predict(X_scaled)
    return df, wcss, optimal_k

def generate_cluster_titles(df, client):
    cluster_titles = {}
    for cluster_id in df['風格'].unique():
        cluster_data = df[df['風格'] == cluster_id]
        avg_feats = cluster_data[['亮度','對比度','飽和度','複雜度','有臉']].mean().to_dict()
        sample_titles = cluster_data['影片標題'].head(5).tolist()  

        prompt = f"""
        你是一位 YouTube 分析專家。根據以下群組影片特徵：
        平均亮度：{avg_feats['亮度']:.1f}，對比度：{avg_feats['對比度']:.1f}，飽和度：{avg_feats['飽和度']:.1f}，
        複雜度：{avg_feats['複雜度']:.1f}%，有臉率：{avg_feats['有臉']*100:.1f}%，
        標題範例：{', '.join(sample_titles)}

        請給這個群組取一個精簡有趣、吸引人的「群標題」，不超過5個字。
        """
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"你是一位專業 YouTube 分析師。"},
                      {"role":"user","content":prompt}]
        )
        cluster_titles[cluster_id] = completion.choices[0].message.content.strip()
    return cluster_titles

# ==========================================
# 4. Streamlit 介面
# ==========================================
st.set_page_config(page_title="YouTube 縮圖標題醫生", layout="wide")

with st.sidebar:
    st.title("🎯 診斷控制台")
    user_topic = st.text_input("搜尋主題", "iPhone 16 開箱")
    user_title = st.text_input("預計標題", "這支手機真的值得買嗎？")
    num_videos = st.slider("樣本數量", 20, 50, 30)
    st.divider()
    uploaded_file = st.file_uploader("上傳你的縮圖", type=["jpg","png","jpeg"])
    start_analysis = st.button("🚀 執行深度分析")

st.title("🩺 YouTube 爆紅基因診斷室")

if start_analysis and user_topic and uploaded_file and user_title:
    with st.spinner(f"正在分析「{user_topic}」的市場爆紅基因..."):
        try:
            # 市場資料抓取（長影片 >3分鐘）
            youtube = get_youtube_service()
            search_res = youtube.search().list(
                q=user_topic, type="video", part="id,snippet", maxResults=num_videos, order="viewCount",
                videoDuration="medium"
            ).execute()
            v_ids = [item['id']['videoId'] for item in search_res['items']]
            v_stats = youtube.videos().list(
                id=','.join(v_ids), part="snippet,statistics,contentDetails"
            ).execute()

            market_records = []
            for v in v_stats['items']:
                thumb_url = v['snippet']['thumbnails'].get('high', v['snippet']['thumbnails'].get('default'))['url']
                resp = requests.get(thumb_url).content
                vision_feats = analyze_advanced_vision(np.frombuffer(resp, np.uint8))
                title_feats = analyze_title_features(v['snippet']['title'])
                if vision_feats:
                    duration = v['contentDetails']['duration']
                    duration_sec = int(pd.to_timedelta(duration).total_seconds())
                    video_url = f"https://www.youtube.com/watch?v={v['id']}"
                    market_records.append(
                        vision_feats + title_feats + [int(v['statistics'].get('viewCount',0)), duration_sec, video_url, v['snippet']['title']]
                    )

            df = pd.DataFrame(market_records, columns=[
                '亮度','對比度','飽和度','複雜度','有臉',
                '標題長度','標題含數字','標題含標點','觀看數','影片秒數','影片連結','影片標題'
            ])

            # 分群
            df, wcss, optimal_k = calculate_clustering_logic(df)
            cluster_titles = generate_cluster_titles(df, client)
            df['風格標題'] = df['風格'].map(cluster_titles)

            # 市場趨勢圖表
            st.subheader("📊 市場趨勢分析")
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                st.markdown("**Elbow Method**：橫軸 K，縱軸 WCSS，轉折點可判斷最佳分群數。")
                fig_elbow = px.line(x=range(1,len(wcss)+1),y=wcss,title="WCSS 轉折點分析 (Elbow)",labels={'x':'K','y':'WCSS'})
                fig_elbow.update_traces(mode='lines+markers')
                st.plotly_chart(fig_elbow,use_container_width=True)

            with col_viz2:
                st.markdown("**市場視覺氣泡圖**：X=對比度，Y=飽和度，氣泡=觀看數，顏色=分群風格。")
                fig_bubble = px.scatter(
                    df,x='對比度',y='飽和度',size='觀看數',color='風格標題',
                    title="市場視覺分佈 (氣泡大小=觀看數)"
                )
                st.plotly_chart(fig_bubble,use_container_width=True)

            # 個人診斷
            user_img_bytes = uploaded_file.read()
            user_vision = analyze_advanced_vision(np.frombuffer(user_img_bytes, np.uint8))
            user_title_info = analyze_title_features(user_title)

            st.divider()
            st.subheader("🩺 診斷報告：你 vs 市場平均")
            diag_col1, diag_col2 = st.columns([1.5,1])
            with diag_col1:
                categories = ['亮度','對比度','飽和度','複雜度','有臉']
                m_avg_v = df[['亮度','對比度','飽和度','複雜度','有臉']].mean().values
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=user_vision,theta=categories,fill='toself',name='你的縮圖',line_color='red'))
                fig_radar.add_trace(go.Scatterpolar(r=m_avg_v,theta=categories,fill='toself',name='市場平均',line_color='blue'))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,255])),width=600,height=600,title="視覺特徵雷達對比")
                st.plotly_chart(fig_radar,use_container_width=True)

            with diag_col2:
                comparison_df = pd.DataFrame({
                    "指標":["縮圖亮度","對比度","飽和度","視覺複雜度","縮圖含人臉","標題長度","標題含數字","標題含標點"],
                    "你的數值":[f"{user_vision[0]:.1f}",f"{user_vision[1]:.1f}",f"{user_vision[2]:.1f}",f"{user_vision[3]:.1f}%","是" if user_vision[4] else "否",
                                f"{user_title_info[0]} 字","是" if user_title_info[1] else "否","是" if user_title_info[2] else "否"],
                    "市場平均":[f"{df['亮度'].mean():.1f}",f"{df['對比度'].mean():.1f}",f"{df['飽和度'].mean():.1f}",f"{df['複雜度'].mean():.1f}%",
                               f"{df['有臉'].mean()*100:.1f}%","{:.1f} 字".format(df['標題長度'].mean()),
                               "{:.1f}%".format(df['標題含數字'].mean()*100),"{:.1f}%".format(df['標題含標點'].mean()*100)]
                })
                st.table(comparison_df)

            # 原始資料
            st.divider()
            st.subheader("📄 市場原始資料")
            st.dataframe(df.rename(columns={
                '亮度':'亮度','對比度':'對比度','飽和度':'飽和度','複雜度':'視覺複雜度','有臉':'含人臉',
                '標題長度':'標題長度','標題含數字':'標題含數字','標題含標點':'標題含標點',
                '觀看數':'觀看數','影片秒數':'影片長度(秒)','連結':'影片連結','影片標題':'影片標題','風格標題':'分群風格'
            }))

            # AI 建議
            st.divider()
            st.subheader("🤖 AI 營運專家建議")
            ai_spinner = st.empty()
            ai_spinner.info("AI 正在分析中，請稍候...")
            prompt = f"""
            你是一位 YouTube 增長專家，專注於縮圖與標題優化。
            主題：{user_topic}

            市場平均：
            對比度 {df['對比度'].mean():.1f}，飽和度 {df['飽和度'].mean():.1f}，複雜度 {df['複雜度'].mean():.1f}%，有臉率 {df['有臉'].mean()*100:.1f}%
            標題平均長度 {df['標題長度'].mean():.1f} 字，含數字 {df['標題含數字'].mean()*100:.1f}%，含標點 {df['標題含標點'].mean()*100:.1f}%

            使用者：
            縮圖亮度 {user_vision[0]:.1f}，對比 {user_vision[1]:.1f}，飽和度 {user_vision[2]:.1f}，複雜度 {user_vision[3]:.1f}%，有臉 {"有" if user_vision[4] else "無"}
            標題：{user_title}，長度 {user_title_info[0]}，含數字 {"是" if user_title_info[1] else "否"}，含標點 {"是" if user_title_info[2] else "否"}

            請給出 3 個具體可執行的優化建議：
            1) 對縮圖的亮度、對比、飽和度、複雜度與人臉使用
            2) 對標題的長度、數字與標點使用
            3) 語氣可以犀利但務必具體
            """
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"你是一位專業 YouTube 分析師，給出具體可執行建議，繁體中文回答。"},
                          {"role":"user","content":prompt}]
            )
            ai_spinner.empty()
            st.info(completion.choices[0].message.content)

        except Exception as e:
            st.error(f"分析失敗：{str(e)}")
else:
    st.info("💡 準備就緒！請在左側輸入主題並上傳縮圖。")
