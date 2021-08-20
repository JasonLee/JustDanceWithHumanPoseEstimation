
import bigbang from '../images/BIGBANG.jpeg';
import image from '../images/img_avatar.png';
import dynamite from '../images/dynamite.png';
import queen from '../images/queen.jpg';
import leslie from '../images/leslie.jpg';
import graduation from '../images/graduation.jpg';
import twice from '../images/twice.png';
import M from '../images/M.jpg';
import bp from '../images/blackpink.png'

const ImageService = {
    getImagebyID(imageName) {
        if (imageName == 1) {
            return bp
        }else if(imageName == 2) {
            return dynamite
        }else if(imageName == 3) {
            return bigbang
        }else if(imageName == 4) {
            return queen
        }else if(imageName == 5) {
            return twice
        }else if(imageName == 6) {
            return leslie
        }else if(imageName == 7) {
            return M
        }else if(imageName == 8) {
            return dynamite
        }else {
            return image
        }

    }
}

export default ImageService;