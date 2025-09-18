import cmath
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import os
from pathlib import Path
import time
import concurrent.futures


def cart2pol(row, col):
    '''
    Cartesian to polar coordinates, r = √ ( x2 + y2 ), θ = tan-1 ( y / x )
    :param row: x-coordinate
    :param col: y-coordinate
    :return:log-radius, theta_degree
    '''

    z = complex(row, col)
    rho, theta = cmath.polar(z)
    theta_degrees = np.degrees(theta) + 180
    if rho == 0:
        return 0, theta_degrees
    return np.log(rho), theta_degrees


def cart2polar(ssd_region_size):
    '''
    get polar radius and polar angle
    :param region_size: each coordinate in the region is transformed to a polar coordinate
    :return:radius array, angle array
    '''
    radius = np.zeros(ssd_region_size, dtype=float)  # Polar coordinates radius
    angle = np.zeros(ssd_region_size, dtype=float)   # Polar coordinates angle
    center = (ssd_region_size[0] // 2, ssd_region_size[1] // 2)

    for row in range(0, ssd_region_size[0]):
        for col in range(0, ssd_region_size[1]):
            # Cartesian to polar coordinates
            rho,  theta = cart2pol(row - center[0], col - center[1])
            radius[row, col] = rho
            angle[row, col] = theta

    return radius, angle


def get_bin(radius, angle, ssd_region_size, bin_size):
        '''
        iterate the region，and locate the positions that belong to the same bin and save them to a bins list
        :param radius: radius array
        :param angle: angle array
        :param region_size:
        :param a:number of rows for the bins
        :param b:number of columns for the bins
        :return: bins list that contains the separation for image positions
        '''
        a = bin_size[0]
        b = bin_size[1]
        out_size = ssd_region_size
        max_radius = np.max(radius)  # Maximum radius
        bins = [[[] for _ in range(b)] for _ in range(a)]
        bin_angle = 360 / a
        for m in range(a):
            theta_low, theta_up = m * bin_angle, (m + 1) * bin_angle
            for n in range(b):
                rho_low, rho_up = max_radius * n / b, max_radius * (n + 1) / b
                temp = []

                for row in range(out_size[0]):
                    for col in range(out_size[1]):
                        if (rho_low <= radius[row, col] <= rho_up) and (theta_low <= angle[row, col] <= theta_up):
                            temp.append([row, col])
                bins[m][n] = temp

        return bins


def get_self_sim_vec(ssd_region, bins):
        '''
        generate the self-similarity descriptor
        :param ssd_region:correlation surface being each calculated by the sum of square differences between patchs
        :param bin:
        :param vec_size:the size of self-similarity descriptor
        :return: self-similarity descriptor
        '''
        #print(ssd_region)
        num_features = len(bins) * len(bins[0])
        self_similarities_vec = np.zeros(num_features)

        for m, row_bins in enumerate(bins):
            for n, temp in enumerate(row_bins):
                #print("temp", temp)
                max_value = 0
                for loc in temp:
                    #print("loc", loc)
                    #print(len(ssd_region) // 2, len(ssd_region[0]) // 2)
                    if loc != [len(ssd_region) // 2, len(ssd_region[0]) // 2]:
                        max_value = max(ssd_region[loc[0], loc[1]], max_value)
                        #print("max_value", max_value)

                self_similarities_vec[m * len(bins[0]) + n] = max_value
        #print(self_similarities_vec)
        return self_similarities_vec


def cal_ssd(patch, region, gap):
        '''
        :param patch: the center patch
        :param region:
        :param alpha: the maximal variance of the difference of all patches within a very small neighborhood of q (of radius 1) relative to the patch centered at q.
        :param center_patch:size of patch
        :return:
        '''
        region_size = region.shape
        # Calculate output dimensions without padding
        output_height = (region_size[0] - len(patch))//gap[0] + 1
        output_width = (region_size[1] - len(patch[0]))//gap[1] + 1
        #print("output_height, output_width", output_height, output_width)
        SSD_region = np.zeros((output_height, output_width))

        for row in range(0, output_height*gap[0], gap[0]):
            for col in range(0, output_width*gap[1], gap[1]):
                  temp = region[row:row+len(patch), col:col+len(patch[0])] - patch[:, :]
                  SSD_region[row//gap[0], col//gap[1]] = np.sum(temp**2)
                  SSD_region[row//gap[0], col//gap[1]] = np.exp(-SSD_region[row//gap[0], col//gap[1]])

        return SSD_region


def mapminmax(vec, min, max):
    '''

    :param vec:vec to be normalized
    :param min: scale
    :param max: scale
    :return: normalized output
    '''
    # Initialize the MinMaxScaler
    vec = vec.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(min, max))
    # Fit and transform the vector
    LSSD = scaler.fit_transform(vec)

    return LSSD

def process_subregion(sub):
    #print(sub.shape)
    middle_index = len(sub) // 2

    # Extract the middle 3 numbers
    middle_three = sub[middle_index - 1:middle_index + 2].reshape(1, -1)
    region = sub.reshape(1, -1)
    ssd_region = cal_ssd(middle_three, region, [1, 1])
    #print(ssd_region.shape)
    radius, angle = cart2polar(ssd_region.shape)
    #print(radius, angle)
    bins = get_bin(radius, angle, ssd_region.shape, [1, 3])
    vec = get_self_sim_vec(ssd_region, bins)
    LSSD = mapminmax(vec, 0, 1).flatten()
    #print(LSSD)
    return LSSD

def getImgWidth(filesize):
    kb = 1024
    # Example usage:
    file_size_ranges = {
        (0, 10*kb): 32,
        (10*kb, 30*kb): 64,
        (30*kb, 60*kb): 128,
        (60*kb, 100*kb): 256,
        (100*kb, 200*kb): 384,
        (200*kb, 500*kb): 512,
        (500*kb, 1000*kb): 768,
        (1000*kb, 2000*kb): 1024,
        (2000*kb, 5000*kb): 1536,
        (5000*kb, 100000000000*kb): 2048,
    }
    for size_range, width in file_size_ranges.items():
        start, end = size_range
        if start <= filesize <= end:
            return width
    print("no range is found")


def readbinary(file_path):
    try:
        with open(file_path, 'rb') as file:
            # Read the entire file into a binary variable
            binary_data = file.read()

    except FileNotFoundError:
        print(f'The file "{file_path}" was not found.')

    except Exception as e:
        print(f'An error occurred: {e}')
    return binary_data


def parallel_windows_maps(srcdir, outdir):
    execution_time = 0
    os.makedirs(outdir, exist_ok=True)
    for root, dirs, files in os.walk(srcdir):
        for subdirectory in dirs:
            subdirectory_path = os.path.join(root, subdirectory)
            destination_dir = os.path.join(outdir, subdirectory)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            # Check if the subdirectory is not empty
            if os.listdir(subdirectory_path):
                # Iterate through files in the subdirectory
                for filename in os.listdir(subdirectory_path):
                    source_filepath = os.path.join(subdirectory_path, filename)
                    destination_path = os.path.join(destination_dir, filename+'.png')
                    if Path(source_filepath).exists():
                        print(source_filepath)
                        start_time = time.time()
                        binary_data = readbinary(source_filepath)
                        byte_array = np.frombuffer(binary_data, dtype=np.uint8)
                        image = np.array(byte_array).flatten()

                        number_regions = (image.shape[0] - 15) // 15 + 1
                        subregions = np.array([process_subregion(image[i:i + 15]) for i in range(0, number_regions*15, 15)])
                        normalized_array = ((subregions - subregions.min()) / (subregions.max() - subregions.min()) * 255).astype(np.uint8).reshape(-1)
                        width = getImgWidth(normalized_array.nbytes)
                        height = normalized_array.shape[0]//(width*3)
                        normalized_array = normalized_array[:width * height * 3].reshape(-1, 3).reshape(width, -1, 3)
                        #print(normalized_array.shape)
                        image = Image.fromarray(normalized_array)
                        end_time = time.time()
                        per_execution = (end_time - start_time)
                        execution_time+=per_execution
                        image.save(destination_path)
    execution_time = execution_time/9339
    print(execution_time)




