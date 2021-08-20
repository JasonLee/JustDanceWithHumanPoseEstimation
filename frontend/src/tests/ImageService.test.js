import ImageService from '../services/ImageService'
import bp from '../images/blackpink.png'

describe('ImageService Test', () => {
    it("should return an image", () => {
        const image = ImageService.getImagebyID(1);
        // expect(image).toBeInstanceOf(Image);
        expect(image).toBe(bp);
    });

});