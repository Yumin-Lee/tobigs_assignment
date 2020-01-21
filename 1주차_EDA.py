#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# #-*- coding: utf-8 -*-


# In[2]:


import pandas as pd
import numpy as np
import csv
import seaborn as sns; sns.set
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


pd.set_option('max_columns', 24,'max_rows', 38)


# # 데이터 불러오기

# In[4]:


df = pd.read_csv('Auction_master_train_1.csv', encoding='cp949')


# In[5]:


df.head(25)


# # 1. 결측치 제거

# In[6]:


df.isnull()


# In[7]:


df2 = df.copy()


# In[8]:


print(df2.dropna(axis=1).shape)


# In[9]:


df2.isnull().sum(axis=0)


# # 2. 상관관계 히트맵
# 
# Hammer_price와 각 변수 사이의 상관관계 show

# In[10]:


sns.heatmap(df2.corr(),cmap='summer')


# # 3. 분포 히스토그램

# In[11]:


df2.hist(bins=30,figsize=(20,20))


# # 4. 시각화

# In[12]:


df2.columns #데이터프레임 열 이름 나열


# # 4-1. hammer price의 분포

# In[13]:


df2['Hammer_price'].plot.hist(bins=10,color='grey',edgecolor='black')
plt.title('낙찰가')
plt.show()


# # 4-2. 낙찰가-총감정가 산점도

# In[15]:


plt.scatter(x='Total_appraisal_price', y='Hammer_price', data=df)
plt.show()


# In[16]:


# 이상치 제거
df_3 = df2.query('Total_appraisal_price < 1e10 & Hammer_price <1e10')
plt.scatter(x='Total_appraisal_price', y='Hammer_price', data=df_3)
plt.show()


# In[17]:


#한글 깨짐 현상 해결
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


# # 4-3. pie-chart

# In[18]:


bid_class = df2['Bid_class'].value_counts()


# In[19]:


bid_class = bid_class.rename(index={"일반":0})


# In[20]:


bid_class = bid_class.rename(index={"개별":1})


# In[21]:


bid_class = bid_class.rename(index={"일괄":2})


# In[22]:


bid_class #index명 변경한 bid_class


# In[23]:


# Pie-chart
bid_class.plot.pie(explode=[0,0.1, 0.2],autopct='%1.2f%%',shadow=True, labels =['일반','개별','일괄'],colors =['pink','skyblue','yellow']) 
plt.show()


# # 4-4.경매구분(강제/일반)

# In[24]:


sns.countplot("Auction_class" ,data=df2)
plt.show()


# # 4-5.count plot 

# In[25]:


sns.countplot("addr_do", hue='addr_si', data=df2)
plt.legend(loc='right',bbox_to_anchor=(1.5,1))
plt.show() 


# # 5. 피처 생성

# # 5-1 현재 층수 기준 저/중/고층 피처 생성

# In[27]:


def floor(x):
    if 0 <= x <= 10:
        return '저층'
    elif 11 <= x <= 20:
        return '중층'
    else:
        return '고층'

df2['층수'] = df2['Current_floor'].apply(floor)


# # 5-2.  날짜 피처 생성

# In[44]:


df2['First_auction_date'].head()


# In[45]:


df2['year'] = df2['First_auction_date'][0:3]


# In[ ]:


# 5-3. 날짜 피처 생성
def year(x):
    if '2014-01-01 0:00' <= x < '2015-01-01 0:00':
        return '2014년'
    elif '2015-01-01 0:00' <= x < '2016-01-01 0:00':
        return '2015년'
    elif '2016-01-01 0:00' <= x < '2017-01-01 0:00':
        return '2016년'
    elif '2017-01-01 0:00' <= x < '2018-01-01 0:00':
        return '2017년'
    else:
        return '확인 요망'

df2['year'] = df2['Appraisal_date'].apply(year)


# 날짜 피처를 생성하고 싶었는데 연도만 분리하는 데에 문제가 생겨 피처를 완성하지 못했습니다.

# # 5-3. 피처 생성

# In[26]:


g = sns.kdeplot(df2.query("Bid_class == '일반'")['Hammer_price'], color = "red",shade= True)
g = sns.kdeplot(df2.query("Bid_class == '개별'")['Hammer_price'], color = 'blue', shade = True)
g = sns.kdeplot(df2.query("Bid_class == '일괄'")['Hammer_price'], color = 'yellow', shade = True)

g.set_ylabel('비율')
g =g.legend(['일반','개별','일괄'])
plt.show()


# # 5-4 총경매횟수 피처생성

# In[29]:


def count_auc(x):
    if 0 <= x <= 1:
        return '1회'
    elif 2 <= x <= 3:
        return '2-4회'
    else:
        return '5회 이상'

df2['count_auc'] = df2['Auction_count'].apply(count_auc)


# # 5-5 건물 총 층수 피처 생성

# In[30]:


def tot_floor(x):
    if 0 <= x <= 10:
        return '저층'
    elif 11 <= x <= 20:
        return '중층'
    else:
        return '고층'

df2['총층수'] = df2['Total_floor'].apply(tot_floor)


# # 5-6. 실면적/전체면적 비율 피처 생성

# In[31]:


# 대지활용도=실면적/전체면적 열 추가

df2['대지활용비율'] = df2['Total_land_real_area'] / df2['Total_land_gross_area']


# In[32]:


df2['대지활용비율'].head(20)


# In[33]:


df2['대지활용비율'].plot.hist(bins=10,color='grey',edgecolor='black',
                        range=[0,0.2])


# In[34]:


def area_ratio(x):
    if 0 <= x <= 0.025:
        return '낮음'
    else:
        return '높음'

df2['대지활용도'] = df2['대지활용비율'].apply(area_ratio) #피처 생성


# # 5-7. Good Deal 피처 생성

# In[35]:


# 총감정가 대비 낙찰가 열 추가

df2['총감정가_낙찰가'] = df2['Total_appraisal_price'] / df2['Hammer_price']


# In[36]:


df2['총감정가_낙찰가'].plot.hist(bins=10,color='grey',edgecolor='black',
                         range=[0,3])
plt.title('총감정가_낙찰가')
plt.show()


# # 5-8 Good_Deal 피처 생성

# In[37]:


# 5-8 Good_Deal 피처 생성
def deal(x):
    if 0 <= x <= 1:
        return 'Bad Deal'
    elif x == 1:
        return '적정'
    else:
        return 'Good Deal'

df2['deal'] = df2['총감정가_낙찰가'].apply(deal)


# # 5-9 유찰비율 피처 생성

# In[38]:


df2['유찰비율'] = df2['Auction_miscarriage_count'] / df2['Auction_count']


# In[39]:


df2['유찰비율'].plot.hist(bins=10,color='grey',edgecolor='black')
plt.title('유찰비율')
plt.show()


# In[40]:


def popularity(x):
    if 0 <= x <= 0.3:
        return '좋은 매물'
    elif 0.3 <= x <= 0.6:
        return '일반 매물'
    else:
        return '나쁜 매물'

df2['ratio_유찰'] = df2['유찰비율'].apply(popularity)


# # 5-10. 경매 신청인의 만족도 피처 생성

# In[41]:


df2['낙찰가/청구가'] = df2['Hammer_price'] / df2['Claim_price']


# In[42]:


df2['낙찰가/청구가'].plot.hist(bins=10,color='grey',edgecolor='black',
                         range=[0,3])
plt.title('낙찰가/청구가')
plt.show()


# In[43]:


#만족도 피처 생성
def satisfaction(x):
    if 0 <= x <= 1:
        return '불만족'
    elif x == 1:
        return '적정'
    else:
        return '만족'

df2['주인_satis'] = df2['낙찰가/청구가'].apply(satisfaction)

