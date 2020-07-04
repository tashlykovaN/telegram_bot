import logging
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import os

PORT = int(os.environ.get('PORT', 5000))
TOKEN = '1178906656:AAFXJ-JiWxZLuwfCJmj4usMjJ4-_Kgve4pw'

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

bot = telegram.Bot(token=TOKEN)

FLAG_NUM_PHOTO = 1
content_img_name = '0'
style_img_name = '0'


def start(update, context):
    update.message.reply_text('Привет! Рада видеть вас здесь:)Я могу переделывать твои фотографии в любой стиль, какой мне покажешь.')
    update.message.reply_text('Отправь мне картику, которую хотел бы поменять!')


def help(update, context):
    update.message.reply_text('Я могу переделывать твои фотографии в любой стиль, какой мне покажешь.')


def echo(update, context):
    update.message.reply_text(update.message.text)


def photo_handler(update, context):

    global FLAG_NUM_PHOTO
    if FLAG_NUM_PHOTO == 1:
        content_img = bot.getFile(update.message.photo[-1].file_id)
        content_img.download('user_content_img.jpg')
        global content_img_name
        content_img_name = str(content_img['file_id'])
        update.message.reply_text('Теперь загрузи картинку, стиль которой мне нужно взять!')
        #.message.reply_text(f'{type(content_img_name)}{str(content_img_name)}')
    else:
        style_img = bot.getFile(update.message.photo[-1].file_id)
        style_img.download('user_style_img.jpg')
        global style_img_name
        style_img_name = str(style_img['file_id'])
        update.message.reply_text('Извини, я только учусь переделывать картинки. На данный момент у меня не получается. Возвращаю тебе твои фотографии)')
        bot.send_photo(update.message.from_user.id, photo=style_img_name)
        bot.send_photo(update.message.from_user.id, photo=content_img_name)

    FLAG_NUM_PHOTO *= -1


def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():

    updater = Updater(TOKEN, use_context=True)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    dp.add_handler(MessageHandler(Filters.text, echo))
    dp.add_handler(MessageHandler(Filters.photo, photo_handler))

    dp.add_error_handler(error)

    updater.start_webhook(listen="0.0.0.0",
                          port=int(PORT),
                          url_path=TOKEN)
    updater.bot.setWebhook('https://immense-ocean-60357.herokuapp.com/' + TOKEN)

    updater.idle()


if __name__ == '__main__':
    main()
