import image from '../logo.svg';
import bigbang from '../images/BIGBANG.jpeg';

const ImageService = {
    getImage(imageName) {
        if (imageName == "TWICE") {
            return bigbang
        }

    }
}

export default ImageService;