#!/usr/bin/env python
# coding: utf-8

# # 문제 상황

# Input data(합성 데이터)
# - 고객명
# - 고객 e-mail
# - 국가
# - 성별
# - 나이
# - 연봉 
# - 빚 
# - 순자산
# 
# Output data
# - 자동차 구매 지불 능력(금액)

# # STEP #0: 라이브러리 입력
# 

# In[1]:


import pandas as pd # 테이블 조작에 사용
import numpy as np # 수치 해석에 사용
import matplotlib.pyplot as plt # 데이터 시각화에 사용
import seaborn as sns # '' 


# # STEP #1: 데이터 셋 입력

# In[2]:


car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')


# In[3]:


car_df


# # STEP #2: 데이터 시각화

# In[4]:


sns.pairplot(car_df) # seaborn pairplot -> 데이터의 행렬 정보 보여줌


# # STEP #3: 데이터 정규화

# In[5]:


X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)


# 쓸모없는 데이터 제거(Drop) - 이름, 이메일, 국가, 차량 구매 금액
# 입력값(X)은 반드시 대문자로 작성 

# In[6]:


X


# In[7]:


y = car_df['Car Purchase Amount']
y.shape


# In[8]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X) # X 데이터 정규화


# In[9]:


scaler.data_max_ # MinMaxScaler 시각화(max) 


# In[10]:


scaler.data_min_ # MinMaxScaler 시각화(min) 


# In[11]:


print(X_scaled[:,0])


# In[12]:


y.shape


# In[13]:


y = y.values.reshape(-1,1) # y 데이터 재구조화


# In[14]:


y.shape


# In[15]:


y_scaled = scaler.fit_transform(y) # y 데이터 정규화


# In[16]:


y_scaled


# # STEP#4: 데이터 트레이닝 & 테스트

# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)


# In[18]:


X_scaled.shape


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[21]:


import tensorflow.keras # 신경망 API keras 불러오기 
from keras.models import Sequential # 신경망을 순차적 형태로 설계하기 위한 import
from keras.layers import Dense # 완전 연결 신경망을 사용하기 위한 import
from sklearn.preprocessing import MinMaxScaler

model = Sequential() # 순차적으로 은닉층 추가 
model.add(Dense(25, input_dim=5, activation='relu')) # 뉴런의 개수, dimension 개수, 활성함수 선언 
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


# In[22]:


model.compile(optimizer='adam', loss='mean_squared_error') # adam 알고리즘 사용, 손실함수 정의


# In[23]:


epochs_hist = model.fit(X_train, y_train, epochs=20, batch_size=25,  verbose=1, validation_split=0.2) # 20번 에포크, 배치 25, 


# # STEP#5: 모델 평가

# In[24]:


print(epochs_hist.history.keys())


# In[25]:


plt.plot(epochs_hist.history['loss']) # matplot 라이브러리로 데이터 시각화
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])


# In[35]:


# 성별, 나이, 연봉, 카드빚, 순자산

X_Testing = np.array([[1, 50, 50000, 10000, 100000]])


# In[36]:


y_predict = model.predict(X_Testing)
y_predict.shape


# In[37]:


print('예상 구매 가능 금액 =', y_predict[:,0])

