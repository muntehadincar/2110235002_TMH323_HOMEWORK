#!/usr/bin/env python
# coding: utf-8

# # Body Fat Prediction
# ## 1. Veri seti hakkında bilgi edinme süreci
# ### Dataset Hakkında
# 252 erkekten alınmış vücut ölçülerini kullanarak vücut yağlarını hesaplayan veriseti
# #### Girdiler ve çıktılar
# * Density=Su altında ölçülmüş yoğunluk
# * Bodyfat=13 değişkenle elde edilen çıktı(yüzde olarak)
# * Age=Yaş(yıl cinsinden)
# * Weight= Ağırlık(lbs cinsinden)
# * Height=Uzunluk(inches cinsinden)
# * Neck=Boyun ölçüsü(cm cinsinden)
# * Chest=Göğüs ölçüsü(cm cinsinden)
# * Abdomen=Karın ölçüsü(cm cinsinden)
# * Hip=Kalça ölçüsü(cm cinsinden)
# * Thigh=İç bacak ölçüsü(cm cinsinden)
# * Knee=Diz ölçüsü(cm cinsinden)
# * Ankle=Ayak bileği ölçüsü(cm cinsinden)
# * Biceps=Üstteki ön kol ölçüsü(cm cinsinden)
# * Forearm=Ön kol ölçüsü(cm cinsinden)
# * Wrist=El bileği ölçüsü(cm cinsinden)
# #### Bu veriler, "Basit ölçüm tekniklerini kullanan erkekler için genelleştirilmiş bir vücut kompozisyonunun tahmin denklemi", K.W. Penrose, A.G. Nelson, A.G. Fisher, FACSM, İnsan Performansı Araştırma Merkezi, Brigham Young Üniversitesi, Provo, Utah 84602, Tıp ve Spor ve Egzersizde Bilim, cilt. 17, hayır. 2, Nisan 1985, s. 189.

# ## 2. Veri hazırlık süreci
# ## Gerekli kütüphaneleri import edelim.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Veri setini sisteme yükleyelim ve head() komutuyla ilk 5 satıra göz atalım.

# In[2]:


df=pd.read_csv("bodyfat.csv",sep=",")
df.head()


# ## Veri setindeki yağ oranlarının dağılımını incelemek istedim.

# In[3]:


plt.hist(df["BodyFat"],bins=40)
plt.title("Body Fat Histogram Grafiği")
plt.show()


# ## Info() komutuyla veri setini inceleyelim.

# In[4]:


df.info() 


# ## Normalizasyon işlemi yapalım. Bu işlem daha verimli bir sonuç almamızı sağlar.

# In[7]:


from sklearn.preprocessing import MinMaxScaler
# Normalizasyon için scaler nesnesi oluşturma
scaler = MinMaxScaler()
# Girdi verilerini normalleştirme (BodyFat hariç tüm sütunlar)
X_normalized = scaler.fit_transform(df.drop('Density', axis=1))
#Normalleştirilmiş verileri DataFrame'e dönüştürme
X_normalized = pd.DataFrame(X_normalized, columns=df.columns.drop('BodyFat'))
print(X_normalized.head())


# ## Veri setinin son halini "body" isimli bir csv dosyası olarak kaydettim.

# In[8]:


df.to_csv("body.csv",sep=",")


# ## 3. Multiple Linear Regression işlemine geçelim

# In[41]:


from sklearn.linear_model import LinearRegression
df=pd.read_csv("body.csv",sep=",")
df.head()


# ## Değerlerimizi x ve y değişkenlerine atadıktan sonra fit edelim.

# In[42]:


# Girdi ve çıktı değişkenlerini ayırma
X = df.drop('BodyFat', axis=1)
y = df['BodyFat']
from sklearn.model_selection import train_test_split
# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlr=LinearRegression()
mlr.fit(X_train,y_train)


# In[43]:


y_pred = mlr.predict(X_test)


# ## Bunlar aslında b0 ve b1 değerlerimiz

# In[44]:


print(mlr.intercept_,mlr.coef_)


# ## Mutliple linar regression kullanarak r2 score, mae, mse, rmse, mape değerlerini bulalım.

# In[45]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
print("mlr r2:",r2_score(y_test,y_pred))
print("mlr mae:",mean_absolute_error(y_test,y_pred))
print("mlr mse:",mean_squared_error(y_test,y_pred))
print("mlr rmse:",(mean_squared_error(y_test,y_pred))**0.5)
print("mlr mape:",mean_absolute_percentage_error(y_test,y_pred))


# ## 4. Decision Tree Regressor kütüphanesini import edip r2 score, mae,mse,rmse,mape değerlerini görelim.

# In[46]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error


# In[47]:


tree_reg=DecisionTreeRegressor()
tree_reg.fit(X_train,y_train)
tree_pred=tree_reg.predict(X_test)
print("tree r2:",r2_score(y_test,tree_pred))
print("tree mae:",mean_absolute_error(y_test,tree_pred))
print("tree mse:",mean_squared_error(y_test,tree_pred))
print("tree rmse:",(mean_squared_error(y_test,tree_pred))**0.5)
print("tree mape:",mean_absolute_percentage_error(y_test,tree_pred))


# ## 5. Random Forest Regressor kullanarak değerlerimizi predict edelim ve r2 score, mae, mse, rmse, mape değerlerine bakalım.

# In[48]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=37)
rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)
print("rf r2:",r2_score(y_test,rf_pred))
print("rf mae:",mean_absolute_error(y_test,rf_pred))
print("tree mse:",mean_squared_error(y_test,rf_pred))
print("tree rmse:",(mean_squared_error(y_test,rf_pred))**0.5)
print("tree mape:",mean_absolute_percentage_error(y_test,rf_pred))


# ## 6. Son olarak birlikte sonuçları görmek için train test split uygulayalım

# In[49]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=37)

mlr=LinearRegression()
mlr.fit(x_train,y_train)
mlr_pred=mlr.predict(x_test)

dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
dt_pred=dt.predict(x_test)


rf=RandomForestRegressor(n_estimators=100,random_state=37)
rf.fit(x_train,y_train)
rf_pred=rf.predict(x_test)


# In[50]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("mlr: ","r2:",r2_score(y_test,mlr_pred),"mae:",mean_absolute_error(y_test,mlr_pred))
print("dt: ","r2:",r2_score(y_test,dt_pred),"mae:",mean_absolute_error(y_test,dt_pred))
print("rf: ","r2:",r2_score(y_test,rf_pred),"mae:",mean_absolute_error(y_test,rf_pred))


# ## Yorum:

# * En başından itibaren train değerlerle çalıştım ve overfite düşmemiş oldu. Hatanın nereden kaynaklandığını analdığımızda sorunu daha kolay çözebiliyoruz.
# * MSE değeri modelin ortalama kare hatasıdır.Ne kadar az o kadar iyi.MSE=8.80X10-5=0.628
# * RMSE=0.00938(Kök ortalama kare hata)
# * Bu değerler gösteriyor ki düşük bir MSE değeri modelin gerçek değerlere ne kadar yakın tahmin yaptığını gösterir.
# * r2_score, modelin bağımsız değişkenlerinin varyansının %62.8 açıkladığını gösterir.

# ## Birkaç ek bilgi 

# In[51]:


df_errors = pd.DataFrame({'Gerçek': y_test, 'Tahmin': y_pred})
df_errors['Hata'] = df_errors['Gerçek'] - df_errors['Tahmin']


# In[52]:


import matplotlib.pyplot as plt

# Hataların histogramını çizme
plt.hist(df_errors['Hata'], bins=30)
plt.xlabel('Hata')
plt.ylabel('Frekans')
plt.title('Hata Dağılımı')
plt.show()


# In[53]:


df_errors


# In[54]:


hatalar = y_test - y_pred

for feature in X_train.columns:
    plt.scatter(X_test[feature], hatalar)
    plt.title(f"Hata vs {feature}")
    plt.xlabel(feature)
    plt.ylabel('Hata')
    plt.show()


# In[ ]:




