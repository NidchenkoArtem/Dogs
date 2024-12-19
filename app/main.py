from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import base64
from .model import predict

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <style>
                body {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;  /* Высота экрана */
                    margin: 0;
                    font-family: Arial, sans-serif;
                    background-color: #f0f0f0;
                    flex-direction: column;
                }

                .header {
                    text-align: center;
                    font-size: 30px;
                    font-weight: bold;
                    margin-bottom: 40px;
                }

                .container {
                    text-align: center;
                }

                .button {
                    background-color: #4CAF50; /* Зеленый цвет */
                    border: none;
                    color: white;
                    padding: 20px 40px;  /* Увеличены размеры */
                    font-size: 20px;  /* Увеличен шрифт */
                    border-radius: 12px; /* Скругленные углы */
                    cursor: pointer; /* Курсор при наведении */
                    margin: 10px;
                    display: inline-block;
                }

                .file-input {
                    display: inline-block;
                    padding: 15px 30px;  /* Увеличены размеры */
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-size: 20px;  /* Увеличен шрифт */
                    cursor: pointer;
                    text-align: center;
                    margin: 10px;
                }

                .file-input input[type="file"] {
                    display: none;
                }

                #preview {
                    margin-top: 20px;
                    max-width: 300px;
                    max-height: 300px;
                    display: none;
                }

                .footer {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    font-size: 16px;
                    color: #333;
                }
            </style>
            <script>
                function previewImage(event) {
                    const input = event.target;
                    const reader = new FileReader();

                    reader.onload = function() {
                        const preview = document.getElementById('preview');
                        preview.src = reader.result;
                        preview.style.display = 'block'; // Показываем изображение
                    };

                    if (input.files && input.files[0]) {
                        reader.readAsDataURL(input.files[0]);
                    }
                }
            </script>
        </head>
        <body>
            <div class="header">
                Предсказание породы собаки по её изображению
            </div>
            <div class="container">
                <form action="/predict" enctype="multipart/form-data" method="post">
                    <label class="file-input">
                        Выбрать изображение
                        <input name="file" type="file" onchange="previewImage(event)" />
                    </label>
                    <br>
                    <button class="button" type="submit">Загрузить</button>
                </form>
                <img id="preview" alt="Предварительный просмотр изображения">
            </div>

            <div class="footer">
                Выполнил: Нидченко Артём. МКИС32
            </div>
        </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
async def upload_file(file: UploadFile = File(...)):
    # Проверка типа файла
    if not file.content_type.startswith('image/'):
        return HTMLResponse(content="<h3>Ошибка: Пожалуйста, загрузите изображение.</h3>", status_code=400)

    # Получаем предсказание
    prediction = await predict(file)

    # Получаем данные изображения
    image_data = await file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')  # Кодируем в base64

    # Формируем строку для отображения изображения
    image_src = f"data:{file.content_type};base64,{image_base64}"

    return f"""
    <html>
        <head>
            <style>
                body {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    font-family: Arial, sans-serif;
                    background-color: #f0f0f0;
                    flex-direction: column;
                }}

                .header {{
                    text-align: center;
                    font-size: 30px;
                    font-weight: bold;
                    margin-bottom: 40px;
                }}

                .container {{
                    text-align: center;
                }}

                .button {{
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 20px 40px;
                    font-size: 20px;
                    border-radius: 12px;
                    cursor: pointer;
                    margin: 10px;
                }}

                img {{
                    max-width: 300px;
                    max-height: 300px;
                    margin-top: 20px;
                }}

                .footer {{
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    font-size: 16px;
                    color: #333;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                Предсказание породы собаки по её изображению
            </div>
            <div class="container">
                <h2>Предсказание: {prediction}</h2>
                <form action="/" method="get">
                    <button class="button" type="submit">Загрузить другое изображение</button>
                </form>
            </div>

            <div class="footer">
                Выполнил: Нидченко Артём. МКИС32
            </div>
        </body>
    </html>
    """
