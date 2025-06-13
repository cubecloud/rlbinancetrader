glas = 'уеаоэяию'
text = list('Возвращает копию строки, в которой заменены все вхождения указанной строки указанным значением')
change_to = '*'
for i in range(len(text)):
    if text[i] in glas or text[i] in glas.upper():
        text[i] = change_to

print(''.join(text))

import re
text = 'Возвращает копию строки, в которой заменены все вхождения указанной строки указанным значением'
glas = r'[уеаоэяию]'
change_to = '*'  # Заменяем на заданное значение
new_text = re.sub(glas, change_to, text, flags=re.IGNORECASE)
print(new_text)
