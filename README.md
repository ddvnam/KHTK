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
## 2.2 Mô hình lựa chọn
## 2.3 So sánh kết quả
