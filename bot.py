import logging
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import os
from PIL import Image
from torch import mm, tensor
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, BatchNorm2d
from torch.nn.functional import mse_loss
from torch.optim import LBFGS
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.models import vgg19
import copy


PORT = int(os.environ.get('PORT', 5000))
TOKEN = '1178906656:AAFXJ-JiWxZLuwfCJmj4usMjJ4-_Kgve4pw'

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

bot = telegram.Bot(token=TOKEN)


class ContentLoss(Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # это константа. Убираем ее из дерева вычеслений
        self.loss = mse_loss(self.target, self.target)

    def forward(self, input):
        self.loss = mse_loss(input, self.target)
        return input


class StyleLoss(Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
        self.loss = mse_loss(self.target, self.target)

    def gram_matrix(self, input):
        batch_size, h, w, f_map_num = input.size()
        features = input.view(batch_size * h, w * f_map_num)
        G = mm(features, features.t())
        return G.div(batch_size * h * w * f_map_num)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = mse_loss(G, self.target)
        return input


class Normalization(Module):

    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = tensor(mean).view(-1, 1, 1)
        self.std = tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class StyleTransfer:

    def __init__(self, content_img, style_img):
        super(StyleTransfer, self).__init__()
        self.cnn = vgg19(pretrained=True).features.eval()
        self.cnn_normalization_mean = tensor([0.485, 0.456, 0.406])
        self.cnn_normalization_std = tensor([0.229, 0.224, 0.225])
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_img = self.image_format(content_img)
        self.style_img = self.image_format(style_img)

    def image_format(self, image_name):
        imsize = 128
        loader = Compose([
            Resize(imsize),  # нормируем размер изображения
            CenterCrop(imsize),
            ToTensor()])
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)
        return image

    def get_style_model_and_losses(self):
        cnn = copy.deepcopy(self.cnn)
        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std)

        content_losses = []
        style_losses = []

        model = Sequential(normalization)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, ReLU):
                name = 'relu_{}'.format(i)
                layer = ReLU(inplace=False)
            elif isinstance(layer, MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]
        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        optimizer = LBFGS([input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(self, input_img, num_steps=500,
                           style_weight=100000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses()
        optimizer = self.get_input_optimizer(input_img)
        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        input_img.data.clamp_(0, 1)

        return input_img

    def imshow(self, tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)  # функция для отрисовки изображения
        unloader = transforms.ToPILImage()
        image = unloader(image)
        #plt.imshow(image)
        return image

    def operate(self):
        input_img = self.content_img.clone()
        output_tensor = self.run_style_transfer(input_img)
        output = self.imshow(output_tensor)

        return output


def start(update, context):
    update.message.reply_text('Привет! Рада видеть вас здесь:)Я могу переделывать твои фотографии в любой стиль, какой мне покажешь.')
    update.message.reply_text('Отправь мне картику, которую хотел бы поменять!')


def help(update, context):
    update.message.reply_text('Я могу переделывать твои фотографии в любой стиль, какой мне покажешь.')


def echo(update, context):
    update.message.reply_text(update.message.text)


# def echo_photo(update, context):
#     photo_file = update.message.photo[-1].get_file()
#     photo_file.download('user_photo.jpg')
#     # category = 'Photo Provided'
#     # user_data[category] = 'Yes'
#     # logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
#     update.message.reply_text('Great!')
#     update.message.reply_photo(photo_file)


def photo_handler(update, context):
    # update.message.reply_text('first!')

    content_img = bot.getFile(update.message.photo[-1].file_id)
    # update.message.reply_text('second!')
    content_img.download('user_content_img.jpg')
    bot.send_photo(update.message.from_user.id, photo=content_img['file_id'])

    update.message.reply_text('Теперь загрузи картинку, стиль которой мне нужно взять!')

    style_img = bot.getFile(update.message.photo[-1].file_id)
    # update.message.reply_text('third!')
    style_img.download('user_style_img.jpg')
    bot.send_photo(update.message.from_user.id, photo=style_img['file_id'])

    # update.message.reply_text('forth')
    styleTransfer = StyleTransfer(content_img=content_img['file_id'], style_img=style_img['file_id'])
    # update.message.reply_text('5')
    new_image = styleTransfer.operate()
    # update.message.reply_text('6')

    bot.send_photo(update.message.from_user.id, photo=new_image)
    # update.message.reply_text('7')
    # update.message.reply_photo(file)
    # #bot.send_photo(chat_id=update.message.chat.id, photo=file)
    # update.message.reply_text('Bad!')


# def photo(update, context):
#     user = update.message.from_user
#     user_data = context.user_data
#     photo_file = update.message.photo[-1].get_file()
#     photo_file.download('user_photo.jpg')
#     bot.send_photo(update.message.from_user.id, photo=photo_file)
#     category = 'Photo Provided'
#     user_data[category] = 'Yes'
#     logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
#     update.message.reply_text('Great! Is the food halal? Vegetarian? Please type in the dietary specifications of the food.')


def error(update, context):
    """Log Errors caused by Updates."""
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
