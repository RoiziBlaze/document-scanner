Описание программы

Данная программа предназначена для сканирования документов и распознавания содержащегося в нём текста. 
В результате её работы создаётся монохромное изображение документа с применёнными алгоритмами сжатия, 
а также текстовый файл, содержащий весь распознанный текст. Для повышения качества распознавания 
текста, а также для улучшения визуального восприятия документа с точки хрения его перспективы, к 
документу применяется алгоритм перспективного преобразования.<br /><br />

Алгоритм распознавания текста поддерживает русский язык.<br /><br />

Управляющие клавиши:<br /><br />

Клавиша "1" — сфотографировать документ и начать его сканирование;<br />
Клавиша "2" — сбросить текущий снимок и сделать новое фото;<br />
Клавиша "3" — включить/выключить отображение текста;<br />
Клавиша "4" — включить/выключить пороговую обработку;<br />
Клавиша "s" — сохранить файлы в форматах .png/.jpg и .txt;<br />
Клавиша "w" — сохранить файл в формате .webp и .txt;<br />
Клавиша "p" — сохранить файл в формате .pdf;<br />
Клавиша "0" — выход из программы.<br /><br />

Для того, чтобы программный код заработал, следует предварительно загрузить требуемые библиотеки.
Это можно сделать, прописав через терминал следующие команды:<br /><br />

"pip3 install torch torchvision"<br />
"pip install easyocr"<br />
"pip install opencv-python"<br />
"pip install fpdf"<br />
