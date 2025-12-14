# Báo cáo ứng dụng học sâu trong nhận diện đa loài trong phục hồi hệ sinh thái
---
# 1. Giới thiệu bài toán
## 1.1 Bối cảnh
- Sự suy giảm đa dạng sinh học tại các rừng mưa nhiệt đới là một vấn đề cấp bách toàn cầu. Tại thung lũng Magdalena (Colombia), dự án phúc hồi sinh thái tại khu bảo tồn thiên nhiên Silencio
  đang nỗ lực chuyển đồi các vùng đất chăn nuôi gia súc thành rừng tự nhiên và để đánh giá được sự thành công thì cần quá trình theo dõi sự hiện diện của các loài chỉ thị là bắt buộc.

  - Bài toán đặt ra là làm sao xây dựng một hệ thống Học Sâu (Deep Learrning) có khả năng phân tích dữ liệu âm thanh từ các thiết bị giám sát thụ động (PAM - passive acoustic monitoring). Mục tiêu của bài toán không chỉ là nhận diện một loài mà mở rông sang bài toán nhận diện đa loài bao gồm chin, lưỡng cư, các động vật khác trong môi trường tự nhiên.
 
## 1.2 Mục tiêu và Thách thức Kỹ thuật
- Hệ thống cần xử lý đầu vào là các đoạn ghi âm soundscapes liên tục và đưa ra dự đoán xác suất xuất hiện của các loài mỗi 5 giây
  - Ràng buộc tài nguyên tính toán: các mô hình phải cân bằng được sự chính xác và tốc độ để có thể chạy được ở edge devices
  - Đặc trưng dữ liệu phức tạp:
    - Phân phối đuôi dài: Dữ liệu huấn luyện bị mất cân bằng nghiêm trọng giữa các loài phổ biến và các loài hiếm/nguy cấp.
    - Nhiễu nền và Đa nguồn phát: Soundscapes chứa tạp âm môi trường (mưa, gió) và sự chồng lấn âm thanh của nhiều loài cùng lúc.
    - Sự khan hiếm dữ liệu gán nhãn: Phải áp dụng các kỹ thuật học sâu tiên tiến (như Semi-supervised learning hoặc Data Augmentation) để tận dụng dữ liệu chưa gán nhãn.

## 1.3 Ý nghĩa thực tiễn
- Số hóa quy trình Giám sát Phục hồi Sinh thái
Đánh giá định lượng hiệu quả phục hồi: Thay vì dựa vào cảm quan, mô hình cung cấp các chỉ số định lượng về sự quay trở lại của các loài động vật tại các khu vực đang được tái tạo rừng. Sự gia tăng phong phú của các loài chim và lưỡng cư là minh chứng rõ nhất cho sự hồi phục của hệ sinh thái.

Mở rộng quy mô giám sát: Giải pháp cho phép xử lý hàng nghìn giờ ghi âm một cách tự động, giúp các nhà bảo tồn tại FBC (Fundación Biodiversa Colombia) giám sát diện rộng mà không tốn kém nhân lực khảo sát thực địa.

- Bảo tồn các loài Nguy cấp và Ít được nghiên cứu

Phát hiện sớm loài quý hiếm: Các mô hình học sâu, nếu được huấn luyện tốt trên dữ liệu đuôi dài, có khả năng phát hiện những tiếng kêu hiếm gặp mà tai người có thể bỏ sót trong hàng giờ ghi âm nhiễu.

Hiểu biết về hành vi loài: Dữ liệu đầu ra giúp các nhà nghiên cứu hiểu rõ hơn về tập tính, thời gian hoạt động và sự phân bố của các loài động vật trong khu bảo tồn.

- Thúc đẩy công nghệ AI xanh (Green AI)
  
Do giới hạn về phần cứng (CPU inference), giải pháp này thúc đẩy việc nghiên cứu các mô hình học sâu tiết kiệm năng lượng (efficient deep learning). Điều này không chỉ có ý nghĩa trong cuộc thi mà còn mở ra khả năng triển khai các hệ thống AI trực tiếp trên các thiết bị thu âm năng lượng thấp (edge devices) đặt giữa rừng sâu trong tương lai.


# 2. Phương pháp
## 2.1 Tiền xử lý dữ liệu
2.1.1 Tổng quan phương pháp

Trong dự án này, dữ liệu âm thanh đầu vào (raw waveform) không được đưa trực tiếp vào mô hình huấn luyện. Thay vào đó, nhóm đã áp dụng phương pháp chuyển đổi tín hiệu sang miền thời gian - tần số (Time-Frequency domain) dưới dạng Mel Spectrogram. Đây là bước quan trọng nhất để chuyển bài toán từ xử lý tín hiệu âm thanh (Audio Signal Processing) sang bài toán thị giác máy tính, tận dụng sức mạnh của các kiến trúc CNN hiện đại như EfficientNet và ResNet.
2.1.2 Phân tích kỹ thuật: Từ Waveform đến Mel Spectrogram

Quá trình chuyển đổi trải qua 3 giai đoạn chính:

1. Biến đổi Short-Time Fourier Transform (STFT):

Tín hiệu âm thanh gốc (x(t)) là một chuỗi biến thiên biên độ theo thời gian. Dữ liệu này che giấu thông tin quan trọng nhất để phân loại tiếng chim là Tần số. 

Nhóm sử dụng STFT để cửa sổ hóa tín hiệu và áp dụng biến đổi Fourier lên từng đoạn nhỏ. Kết quả thu được là Spectrogram tuyến tính, hiển thị cường độ năng lượng tại mỗi tần số theo thời gian. 

Áp dụng Thang đo Mel (Mel Scale):Spectrogram thông thường biểu diễn tần số theo thang đo tuyến tính. Tuy nhiên, hệ thính giác của con người và loài chim hoạt động theo cơ chế phi tuyến tính. Chúng ta nhạy cảm hơn với sự thay đổi ở dải tần số thấp và kém nhạy cảm hơn ở dải tần số cao.Để mô phỏng đặc tính sinh học này, nhóm ánh xạ tần số (f) sang thang đo Mel (m) theo công thức:

m = 2595 * log_{10}(1 + f/700)


2. Bộ lọc Mel (Mel Filterbank): Dữ liệu được đưa qua một bộ lọc gồm n_mels băng tần (trong bài này là 128). Việc này giúp nén không gian dữ liệu, tập trung độ phân giải vào vùng tần số thấp nơi chứa các đặc trưng chính của tiếng chim, và giảm bớt chiều dữ liệu dư thừa ở tần số cao. 


3. Chuyển đổi Logarit:

Cường độ âm thanh sau khi lọc được chuyển sang đơn vị Decibel (dB) bằng hàm Logarit. Điều này giúp cân bằng dải động (dynamic range), làm nổi bật các tín hiệu tiếng chim nhỏ lẫn trong môi trường ồn.

2.1.3. Tại sao lựa chọn Mel Spectrogram? (So sánh & Đánh giá)
Nhóm lựa chọn Mel Spectrogram thay vì Waveform thuần túy hoặc Spectrogram tuyến tính dựa trên 3 lý do cốt lõi:

Lý do 1: Giải mã đặc trưng ẩn (Feature Decoupling)

So với Waveform: Dữ liệu sóng âm thô rất hỗn loạn và biến thiên pha ngẫu nhiên. Mel Spectrogram mở khóa cấu trúc của âm thanh, biến các tiếng hót thành các hình dạng hình học rõ ràng (ví dụ: đường kẻ sọc, đường cong harmonic). Điều này cho phép mô hình học các đặc trưng bất biến thay vì phải học thuộc lòng các giá trị dao động vô nghĩa.

Lý do 2: Tối ưu hóa cho mô hình CNN

Các kiến trúc như EfficientNet-B0 hay ResNet-18 được thiết kế để tìm kiếm các cạnh, góc, và kết cấu trong hình ảnh. Mel Spectrogram có tính chất "địa phương hóa" tương tự hình ảnh: một tiếng hót cụ thể sẽ chiếm một vùng không gian (pixel) nhất định. Việc chuyển đổi này cho phép nhóm áp dụng kỹ thuật Transfer Learning từ các mô hình đã huấn luyện trên ImageNet cực kỳ hiệu quả.

Lý do 3: Mô phỏng tri giác sinh học 

Tiếng chim được tạo ra để giao tiếp và được nghe bởi chim. Tai của chim có cơ chế lọc tần số tương tự thang đo Mel. Do đó, biểu diễn Mel Spectrogram là cách "nhìn" âm thanh gần gũi nhất với cách mà loài chim cảm nhận, giúp mô hình AI tập trung vào đúng các đặc trưng âm học quan trọng.
## 2.2 Mô hình lựa chọn
Nhóm sử dụng các model để xử lý bài toán trên bao gồm: Resnet18, ResNet50, EfficientNet-B0 và EfficientNet-B1 làm backbone chính để trích xuất đặc trưng từ Mel-spectrograms. Lý do lựa chọn bao gồm:

- ResNet50: Đóng vai trò là một baseline mạnh mẽ và ổn định. Với kiến trúc Residual sâu, mô hình có khả năng học các đặc trưng phức tạp của âm thanh mà không gặp vấn đề biến mất đạo hàm, đảm bảo độ chính xác nền tảng tốt.

- EfficientNet (B0 & B1): Tối ưu hóa sự cân bằng giữa độ chính xác và tài nguyên tính toán đảm bảo có thể phù hợp với ràng buộc tài nguyên tính toán.

  =========================
Đoạn này thêm hình vẽ kiến trúc của resnet với efficientnet vào
  =========================

Kích thước nhỏ gọn giúp tăng tốc độ huấn luyện và suy luận (inference), phù hợp với giới hạn thời gian chạy kernel của Kaggle.

Việc thực nghiệm trên nhiều kiến trúc với độ phức tạp khác nhau giúp nhóm đánh giá và lựa chọn được giải pháp tối ưu nhất đối với ràng buộc tài nguyên tính toán. Mục tiêu là tìm ra điểm cân bằng (trade-off) tốt nhất, đảm bảo mô hình đạt độ chính xác cao nhưng vẫn hoạt động mượt mà trong giới hạn tài nguyên.

## 2.3 So sánh kết quả

so sánh kết quả thì trong folder resnet efficient net có ảnh của nó đấy 
phần so sánh, stats  tại sao một số sample hay bị đoán sai thì có trong folder part3 rồi
