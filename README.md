Code chạy với Python 3.7.4 và các lib sau 

```numpy==1.16.5
sklearn==0.21.3
statsmodels==0.10.1
pandas==0.25.1
tensorflow==2.1.0
keras==2.3.1
xgboost==1.5.0
```

**1. Chạy baseline**

   a. Chạy file ARIMA_new_4paper.py để tạo file phi tuyến ARIMA_residuals1.csv

   b. Chạy file Main_new_4paper.py để dự đoán các kết quả

**2. Chạy dữ liệu chứng khoán VN**
 
a. vào file ARIMA_new_4paper.py dòng 50
```
data = pd.read_csv('601988.SH.csv') 
Đổi thành 
data = pd.read_csv('CTG.csv')
```
xong rồi chạy để tạo file phi tuyến ARIMA_residuals1.csv
b. vào file Main_new_4paper.py dòng 45
```
data1 = pd.read_csv("601988.SH.csv")
đổi thành 
data1 = pd.read_csv("CTG.csv")
```
Xong chạy để lấy kết quả dự đoán.

Làm tương tự với 3 mã chứng khoán còn lại


tham khảo file demo_IS6502_CH201.ipynb để biết thêm cách chạy
