import telebot
import os
from telebot import types
from PIL import Image
import pandas as pd
import glob
import tensorflow as tf

global graph, model
graph = tf.get_default_graph()
from keras.layers import *
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from scipy.misc import imsave
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.resnet50 import preprocess_input

token = "edited 2025.04.25"
URL = "https://api.telegram.org/bot" + token + "/"
new = True
name = False

data_users = {}

count_for_new_photo = 0
bot = telebot.TeleBot(token)
list_of_classes = []
mlp = None

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(64, 64, 3), padding='VALID'))
model.add(Conv2D(48, kernel_size=(3, 3), padding='VALID'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
model.add(Conv2D(16, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
model.add(Conv2D(12, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
model.add(AveragePooling2D(pool_size=(19, 19)))
model.add(Flatten())
model.summary()

def convertImg(img):
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    with graph.as_default():
        vgg_feature = model.predict(img_data)
    return (vgg_feature)


def build_model():
    pics = []
    pics_class = []
    for class_of_signature in list_of_classes:
        filenames = glob.glob(('data/' + class_of_signature + '/*.jpg'))
        filenames = sorted(filenames)
        for name in filenames:
            pics_class.append(class_of_signature)
            img = Image.open(name)
            r_img = np.array(img)
            for i in range(r_img.shape[0]):
                for j in range(r_img.shape[1]):
                    sumq = 0
                    for t in range(r_img.shape[2]):
                        sumq += r_img[i][j][t]
                    if sumq > 430:
                        r_img[i][j][0] = 255
                        r_img[i][j][1] = 255
                        r_img[i][j][2] = 255
            img = Image.fromarray(r_img, 'RGB')
            pics.append(img)

    df = pd.DataFrame({'img': pics, 'y': pics_class})

    for i in range(df.shape[0]):
        df['img'][i] = np.array(convertImg(df['img'][i]))

    li = []
    print('DF shape - ', df.shape)
    for i in range(df.shape[0]):
        li.append(df['img'][i])
    X = np.array(li).reshape(df.shape[0], 12)
    y = np.array(df['y'])
    print(X.shape)
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=8)
    # pca.fit(X)
    # X = pca.transform(X)
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    X_train = X
    y_train = y
    global mlp

    mlp = MLPClassifier(solver='lbfgs', random_state=42, hidden_layer_sizes=[32], activation='tanh', max_iter=1000).fit(
        X_train, y_train)


def pred(img):
    r_img = np.array(img)
    for i in range(r_img.shape[0]):
        for j in range(r_img.shape[1]):
            sumq = 0;
            for t in range(r_img.shape[2]):
                sumq += r_img[i][j][t]
            if sumq > 430:
                r_img[i][j][0] = 255
                r_img[i][j][1] = 255
                r_img[i][j][2] = 255
    img = Image.fromarray(r_img, 'RGB')
    img = np.array(convertImg(img)).reshape(1, 12)
    global mlp
    return mlp.predict(img)


@bot.message_handler(commands=['start', 'go'])
def start(message):
    chat_id = message.chat.id
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    bot_check = types.InlineKeyboardButton(text='Проверить подпись', callback_data='check')
    bot_new = types.InlineKeyboardButton(text='Новый пользователь', callback_data='new')
    bot_end = types.InlineKeyboardButton(text='Закончить', callback_data='end')
    keyboard.add(bot_new, bot_check, bot_end)
    bot.send_message(chat_id, "Здравствуйте, чего вам надо?", reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: True)
def callback(call):
    global new
    global name
    global count_for_new_photo
    call_data = call.data
    chat_id = call.message.chat.id
    mess_id = call.message.message_id
    if call.message:
        if call_data == 'check':
            new = False
            bot.edit_message_text(chat_id=chat_id, message_id=mess_id, text='Отправьте фото')
        elif call_data == 'new':
            new = True
            count_for_new_photo = 0
            name = True
            bot.edit_message_text(chat_id=chat_id, message_id=mess_id, text="Введите информацию о себе")
        elif call_data == 'next':
            bot.edit_message_text(chat_id=chat_id, message_id=mess_id, text='Отправьте фото для дальнейшей регистрации')
        elif call_data == 'end':
            bot.edit_message_text(chat_id=chat_id, message_id=mess_id,
                                  text='Для дальнейшего использование введите команду /start')


@bot.message_handler(content_types="text")
def ask(message):
    global new
    global count_for_new_photo
    global name
    chat_id = message.chat.id
    if name:
        data_users[str(chat_id)] = str(message.from_user.username) + '\n' + str(message.text)
        name = False
        bot.send_message(chat_id, text='Отправьте фотографию')
        new = True
        count_for_new_photo = 0
    elif message.text == "Проверить подпись":
        new = False
        bot.send_message(chat_id, "Отправьте фото")
    elif message.text == "Новый пользователь":
        name = True
        bot.send_message(chat_id, "Введите информацию о себе")
    elif message.text == 'Закончить':
        bot.send_message(chat_id=chat_id, text='Для дальнейшего использование введите команду /start')
    else:
        bot.send_message(message.chat.id, "Простите, не могу обработать ваш запрос")


@bot.message_handler(content_types=['document'])
def handle_docs_file(message):
    chat_id = message.chat.id
    global count_for_new_photo
    if new:
        try:
            file_info = bot.get_file(message.document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)

            try:
                os.makedirs("data/" + str(chat_id), mode=0o777)
                list_of_classes.append(str(chat_id))
                print('list_of_classes-', list_of_classes)

            except OSError:
                pass

            print(message.from_user.username)
            src = 'data/' + str(chat_id) + '/img' + str(count_for_new_photo) + '.jpg'
            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)
            bot.reply_to(message, "Фото добавлено")
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            bot_next = types.InlineKeyboardButton(text='Продолжить отправлять', callback_data='next')
            bot_check = types.InlineKeyboardButton(text='Начать проверять подписи', callback_data='check')
            bot_end = types.InlineKeyboardButton(text='Закночить', callback_data='end')
            keyboard.add(bot_next, bot_check, bot_end)
            bot.send_message(chat_id, text='Хотите продолжить?', reply_markup=keyboard)
            build_model()
            count_for_new_photo += 1
        except Exception as e:
            bot.reply_to(message, e)
    else:
        try:
            file_info = bot.get_file(message.document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            src = 'img' + str(chat_id) + '.jpg'

            with open(src, 'wb') as new_file:

                new_file.write(downloaded_file)

            prediction = pred(Image.open(src))

            print(prediction)

            print(str(prediction[0]))
            print(type(str(prediction[0])))
            print('-------')
            print(data_users.values())
            print('-------')
            bot.send_message(chat_id, text='@' + data_users[str(prediction[0])])
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            bot_check = types.InlineKeyboardButton(text='Продолжить проверять', callback_data='check')
            bot_end = types.InlineKeyboardButton(text='Закончить', callback_data='end')
            keyboard.add(bot_check, bot_end)
            bot.send_message(chat_id, text='Что хотите дальше?', reply_markup=keyboard)
            print('5')
        except Exception as e:
            bot.reply_to(message, e)


@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    chat_id = message.chat.id
    global count_for_new_photo
    if new:
        try:
            file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
            downloaded_file = bot.download_file(file_info.file_path)

            try:
                os.makedirs("data/" + str(chat_id), mode=0o777)
                list_of_classes.append(str(chat_id))
            except OSError:
                bot.send_message(chat_id, "было")

            src = 'data/' + str(chat_id) + '/img' + str(count_for_new_photo) + '.jpg'
            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)
            bot.reply_to(message, "Фото добавлено")
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            bot_next = types.InlineKeyboardButton(text='Продолжить отправлять', callback_data='next')
            bot_check = types.InlineKeyboardButton(text='Начать проверять подписи', callback_data='check')
            bot_end = types.InlineKeyboardButton(text='Закночить', callback_data='end')
            keyboard.add(bot_next, bot_check, bot_end)
            bot.send_message(chat_id, text='Хотите продолжить?', reply_markup=keyboard)
            build_model()
            count_for_new_photo += 1
        except Exception as e:
            bot.reply_to(message, e)
    else:
        try:
            file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
            downloaded_file = bot.download_file(file_info.file_path)

            src = 'img' + str(chat_id) + '.jpg'

            with open(src, 'wb') as new_file:

                new_file.write(downloaded_file)

            prediction = pred(Image.open(src))

            print(prediction)

            print(str(prediction[0]))
            print(type(str(prediction[0])))
            print('-------')
            print(data_users.values())
            print('-------')
            bot.send_message(chat_id, text='@' + data_users[str(prediction[0])])
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            bot_check = types.InlineKeyboardButton(text='Продолжить проверять', callback_data='check')
            bot_end = types.InlineKeyboardButton(text='Закончить', callback_data='end')
            keyboard.add(bot_check, bot_end)
            bot.send_message(chat_id, text='Что хотите дальше?', reply_markup=keyboard)
            print('5')
        except Exception as e:
            bot.reply_to(message, e)


bot.polling(none_stop=True, interval=0)
