import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def evaluate_model():
    print("🔍 EVALUATING MODEL PERFORMANCE...")
    
    # Load model
    model_path = 'model/model_cnn.h5'
    if not os.path.exists(model_path):
        print("❌ Model tidak ditemukan. Jalankan training terlebih dahulu.")
        return
    
    model = load_model(model_path)
    print("✅ Model berhasil dimuat")
    
    # Data generator untuk test set
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(128, 128),
        batch_size=16,
        class_mode='binary',
        shuffle=False  # Penting untuk evaluation
    )
    
    print(f"📊 Test samples: {test_generator.samples}")
    print(f"🎯 Classes: {test_generator.class_indices}")
    
    # Predictions
    print("🔄 Melakukan prediksi pada test set...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n" + "="*50)
    print("📈 MODEL EVALUATION METRICS")
    print("="*50)
    print(f"✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"📥 Recall:    {recall:.4f}")
    print(f"⚖️ F1-Score:  {f1:.4f}")
    
    # Confusion Matrix
    print("\n🔍 CONFUSION MATRIX")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Classification Report
    print("\n📊 DETAILED CLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix - Image Authenticity Detection')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Precision, Recall, F1 per class
    from sklearn.metrics import precision_recall_fscore_support
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(y_true, y_pred)
    
    print("\n🎯 METRICS PER CLASS")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 50)
    print(f"{'Real':<10} {precision_per_class[0]:<10.4f} {recall_per_class[0]:<10.4f} {f1_per_class[0]:<10.4f} {support[0]:<10}")
    print(f"{'Fake':<10} {precision_per_class[1]:<10.4f} {recall_per_class[1]:<10.4f} {f1_per_class[1]:<10.4f} {support[1]:<10}")
    
    # Save results to file
    with open('model/evaluation_results.txt', 'w') as f:
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("=" * 30 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    print(f"\n💾 Results saved to: model/evaluation_results.txt")
    print(f"📊 Confusion matrix saved to: model/confusion_matrix.png")

if __name__ == '__main__':
    evaluate_model()