from strategy import *
import pickle
from skimage.metrics import structural_similarity as ssim

no_reg = []
reg = []
zero_prior = []


def write_value(value):
    with open('maps.pkl', 'wb') as f:
        pickle.dump(value, f)


def read_value():
    with open('maps.pkl', 'rb') as f:
        val = pickle.load(f)
    return val


def calculate_ssim(img1, img2):
    img1 = np.clip(np.array(img1)*255, 0, 255).astype(np.uint8)
    img2 = np.clip(np.array(img2)*255, 0, 255).astype(np.uint8)
    return ssim(img1, img2)


def calculate_mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


if __name__ == "__main__":
    SIZE = 10
    NUM_MAPS = 8
    ITERATION = 1000
    SPARSITY = 0.6
    maps = {}
    for i in range(NUM_MAPS):
        true, pred = strategy1(SPARSITY, i, SIZE, ITERATION)
        no_reg.append((true, pred))
        true, pred = strategy2(SPARSITY, i, SIZE, ITERATION)
        reg.append((true, pred))
        true, pred = strategy2(SPARSITY, i, SIZE, ITERATION, True)
        zero_prior.append((true, pred))
        plt.close()

    psnr_no_reg = list(
        map(lambda imgs: calculate_psnr(imgs[0], imgs[1]), no_reg))
    ssim_no_reg = list(
        map(lambda imgs: calculate_ssim(imgs[0], imgs[1]), no_reg))
    mse_no_reg = list(
        map(lambda imgs: calculate_mse(imgs[0], imgs[1]), no_reg))

    psnr_reg = list(
        map(lambda imgs: calculate_psnr(imgs[0], imgs[1]), reg))
    ssim_reg = list(
        map(lambda imgs: calculate_ssim(imgs[0], imgs[1]), reg))
    mse_reg = list(
        map(lambda imgs: calculate_mse(imgs[0], imgs[1]), reg))

    psnr_zero_prior = list(
        map(lambda imgs: calculate_psnr(imgs[0], imgs[1]), zero_prior))
    ssim_zero_prior = list(
        map(lambda imgs: calculate_ssim(imgs[0], imgs[1]), zero_prior))
    mse_zero_prior = list(
        map(lambda imgs: calculate_mse(imgs[0], imgs[1]), zero_prior))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].plot(psnr_no_reg, label='without reg.')
    ax[0].plot(psnr_reg, label='with reg.')
    ax[0].plot(psnr_zero_prior, label='zero prior')
    ax[0].set_xlabel('ith image')
    ax[0].set_ylabel('PSNR value')
    ax[0].legend()

    ax[1].plot(mse_no_reg, label='without reg.')
    ax[1].plot(mse_reg, label='with reg.')
    ax[1].plot(mse_zero_prior, label='zero prior')
    ax[1].set_xlabel('ith image')
    ax[1].set_ylabel('MSE value')
    ax[1].legend()

    ax[2].plot(ssim_no_reg, label='without reg.')
    ax[2].plot(ssim_reg, label='with reg.')
    ax[2].plot(ssim_zero_prior, label='zero prior')
    ax[2].set_xlabel('ith image')
    ax[2].set_ylabel('SSIM value')
    ax[2].legend()
    plt.savefig('plots/metrics.png', bbox_inches='tight')
    plt.close()

    maps['no_reg'] = no_reg
    maps['reg'] = reg
    maps['zero_prior'] = zero_prior
    write_value(maps)
    print(read_value())
