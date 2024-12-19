import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi import UploadFile
from io import BytesIO
from PIL import Image

model = load_model("dogs_projects.h5")

breeds = [
    "Affenpinscher", "Afghan Hound", "African Hunting Dog", "Airedale", "Akita",
    "Appenzeller", "Basenji", "Basset Hound", "Beagle", "Bearded Collie",
    "Beauceron", "Bedlington Terrier", "Belgian Malinois", "Belgian Sheepdog",
    "Belgian Tervuren", "Bergamasco", "Bernese Mountain Dog", "Bichon Frise",
    "Black and Tan Coonhound", "Black Russian Terrier", "Bloodhound", "Border Collie",
    "Border Terrier", "Boston Terrier", "Bouledogue Francais", "Brittany",
    "Brussels Griffon", "Bull Terrier", "Bulldog", "Bullmastiff", "Cairn Terrier",
    "Cavalier King Charles Spaniel", "Chihuahua", "Chinese Crested", "Chinese Shar-Pei",
    "Chow", "Clumber Spaniel", "Cocker Spaniel", "Collie", "Coonhound",
    "Corgi", "Dachshund", "Dalmatian", "Dandie Dinmont Terrier", "Doberman",
    "Dogue de Bordeaux", "English Bulldog", "English Cocker Spaniel", "English Foxhound",
    "English Setter", "English Springer Spaniel", "English Toy Spaniel", "Entlebucher",
    "Field Spaniel", "Finnish Spitz", "Flat-Coated Retriever", "French Bulldog",
    "German Pinscher", "German Shepherd", "German Shorthaired Pointer",
    "German Wirehaired Pointer", "Giant Schnauzer", "Glen of Imaal Terrier",
    "Goldador", "Golden Retriever", "Goldendoodle", "Gordon Setter", "Great Dane",
    "Great Pyrenees", "Greater Swiss Mountain Dog", "Harrier", "Havanese", "Irish Setter",
    "Irish Terrier", "Irish Water Spaniel", "Irish Wolfhound", "Italian Greyhound",
    "Japanese Chin", "Keeshond", "King Charles Spaniel", "Labrador Retriever",
    "Labradoodle", "Lagotto Romagnolo", "Lakeland Terrier", "Leonberger", "Lhasa Apso",
    "Maltese", "Manchester Terrier", "Mastiff", "Mastin Espanol", "Newfoundland",
    "Norfolk Terrier", "Norwegian Buhund", "Norwegian Elkhound", "Norwegian Forest Cat",
    "Papillon", "Pekingese", "Pembroke Welsh Corgi", "Pit Bull Terrier", "Pointer",
    "Pomeranian", "Poodle", "Portuguese Water Dog", "Saint Bernard", "Saluki",
    "Samoyed", "Schipperke", "Schnauzer", "Scottish Deerhound", "Scottish Terrier",
    "Sealyham Terrier", "Shiba Inu", "Shih Tzu", "Siberian Husky", "Smooth Fox Terrier",
    "Soft-Coated Wheaten Terrier", "Spanish Water Dog", "Spinone Italiano", "Staffordshire Bull Terrier",
    "Standard Schnauzer", "Tibetan Mastiff", "Tibetan Spaniel", "Tibetan Terrier",
    "Toy Poodle", "Vizsla", "Weimaraner", "Welsh Springer Spaniel", "Welsh Terrier",
    "West Highland White Terrier", "Whippet", "Yorkshire Terrier"
]

async def predict(file: UploadFile):
    # Загружаем изображение
    image_data = await file.read()

    print(f'Image data (first 100 bytes): {image_data[:100]}')  # Вывод первых 100 байт

    try:
        image = Image.open(BytesIO(image_data))
    except Exception as e:
        print(f'Error opening image: {e}')  # Отладка: печать ошибки
        raise e  # Повторно выбросить исключение для просмотра в журнале

    # Предобработка изображения
    image = image.resize((224, 224))  # Убедитесь, что размер совпадает с моделью
    image_array = np.array(image) / 255.0  # Нормализация
    image_array = np.expand_dims(image_array, axis=0)

    # Отладка: выводим информацию о предобработанном изображении
    print(f'Image array shape: {image_array.shape}')
    print(f'Image data: {image_array}')

    # Предсказание
    predictions = model.predict(image_array)

    # Отладка: выводим вероятности для всех классов
    print(f'Predictions: {predictions}')

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return breeds[predicted_class_index]