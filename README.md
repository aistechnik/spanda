# spanda

### Как запустить приложение на вашем компьютере

1. Клонировать приложение из репозитория

   ```
   $ git clone https://github.com/aistechnik/spanda.git
   ```

2. Перейти в папку приложения

   ```
   $ cd spanda
   ```

3. Создать виртуальное пространство

   ```
   $ python3 -m venv .venv
   $ source .venv/bin/activate
   ```

4. Установить зависимости

   ```
   $ pip install -r requirements.txt
   ```

5. Запустить приложение

   ```
   $ streamlit run src/Home.py
   ```