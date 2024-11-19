import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score

# 클래스별 테스트 데이터 개수
test_counts = {"negative": 5550, "positive": 2750, "neutral": 8300}

# 클래스별 정확도
accuracies = {"negative": 94.7 / 100, "positive": 92.3 / 100, "neutral": 92.7 / 100}

# 잘못 예측된 비율 계산
error_rates = {key: 1 - value for key, value in accuracies.items()}

# 혼동 행렬 생성
true_labels = []
predicted_labels = []

# 부정(0), 긍정(1), 중립(2)의 실제 및 예측값 생성
for label, count in zip([0, 1, 2], [test_counts["negative"], test_counts["positive"], test_counts["neutral"]]):
    correct = int(count * accuracies[list(accuracies.keys())[label]])  # 정확히 맞춘 개수
    incorrect = count - correct  # 틀린 개수

    # 정확히 맞춘 레이블 추가
    true_labels.extend([label] * correct)
    predicted_labels.extend([label] * correct)

    # 틀리게 예측한 레이블 추가 (임의 분포로 분산)
    incorrect_distribution = [0, 1, 2]
    incorrect_distribution.remove(label)  # 자기 자신 제외

    for incorrect_label in incorrect_distribution:
        predicted_labels.extend([incorrect_label] * (incorrect // 2))
        true_labels.extend([label] * (incorrect // 2))

# 혼동 행렬 계산
cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])
print("Confusion Matrix:")
print(cm)

# F1-Score 계산 및 출력
report = classification_report(true_labels, predicted_labels, target_names=["Negative", "Positive", "Neutral"])
print("\nClassification Report:")
print(report)

# 혼동 행렬 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive", "Neutral"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 클래스별 F1-Score 계산
f1_scores = f1_score(true_labels, predicted_labels, average=None, labels=[0, 1, 2])
categories = ["Negative", "Positive", "Neutral"]

# F1-Score 시각화
plt.bar(categories, f1_scores, color=['red', 'green', 'blue'])
plt.title('F1-Score by Class')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.ylim(0, 1)  # F1-Score는 0~1 사이의 값
plt.show()
