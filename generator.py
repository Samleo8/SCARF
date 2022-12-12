import data.load_DTU as DTU
import torch
import os
import numpy as np
from dataloader import SceneDataset
import imageio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def training_visualization(num_images,
                           cfg,
                           i4d,
                           dataset,
                           epoch,
                           generate_specific_object=True,
                           generate_specific_pose=True):
    # Create log dir and copy the config file
    basedir = cfg.basedir
    expname = cfg.expname

    dataset.render_factor = cfg.vis_render_factor \
        if cfg.vis_render_factor > 0 else cfg.render_factor
    dataloader = dataset.get_loader(num_workers=0)

    # Assert that img and render factor are compatible
    assert dataset.H % dataset.render_factor == 0, \
        f'Image height ({dataset.H}) not divisible by render factor ({dataset.render_factor})'
    assert dataset.W % dataset.render_factor == 0, \
        f'Image width ({dataset.W}) not divisible by render factor ({dataset.render_factor})'

    if generate_specific_object:
        iter = cfg.generate_specific_samples
    else:
        iter = range(num_images)

    if generate_specific_pose:
        pose_iter = cfg.gen_pose
    else:
        pose_iter = ['random']

    renderings = []
    for sample in iter:
        for pose in pose_iter:
            savedir = os.path.join(basedir, expname, 'training_visualization',
                                   f'epoch_{epoch}_{sample}_{pose}')
            img_outpath = os.path.join(savedir, f'rendering.png')
            if os.path.exists(savedir):
                continue
            else:
                os.makedirs(savedir)

            if generate_specific_object:
                dataloader.dataset.load_specific_input = sample
                print(
                    f'generating object {dataloader.dataset.load_specific_input}'
                )

            if generate_specific_pose:
                dataloader.dataset.load_specific_rendering_pose = dataset.cam_path[
                    pose]
                print(f'generating pose {pose}')
            render_data = dataloader.__iter__().__next__()['complete']
            rgb = render_and_save(i4d, dataset, render_data, savedir,
                                  img_outpath, bool(generate_specific_pose))
            renderings.append(rgb)

            dataloader.dataset.load_specific_input = None
            dataloader.dataset.load_specific_rendering_pose = None

    plt.xticks([]), plt.yticks([])
    fig = plt.figure()
    for i, img in enumerate(renderings):
        ax = fig.add_subplot(1, len(renderings), i + 1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img, interpolation='bicubic')

    return fig


def render_pose(cfg, i4d, dataset, epoch, specific_obj, pose):

    # Create log dir and copy the config file
    basedir = cfg.basedir
    expname = cfg.expname

    dataloader = dataset.get_loader(num_workers=0)

    savedir = os.path.join(
        basedir, expname, 'renderings',
        f'{specific_obj}_epoch_{epoch}_renderfactor_{cfg.render_factor}_batch_{cfg.fixed_batch}'
    )
    os.makedirs(savedir, exist_ok=True)

    img_outpath = os.path.join(savedir, f'pose_{pose[0]}.png')
    c2w = pose[1]

    if os.path.exists(img_outpath):
        # Rendering already exists.
        return

    dataloader.dataset.load_specific_input = specific_obj
    dataloader.dataset.load_specific_rendering_pose = c2w
    print(
        f'generating {dataloader.dataset.load_specific_input}, pose: {pose[0]}'
    )
    render_data = dataloader.__iter__().__next__()['complete']

    render_and_save(i4d, dataset, render_data, savedir, img_outpath, True)

    dataloader.dataset.load_specific_input = None
    dataloader.dataset.load_specific_rendering_pose = None


def render_and_save(i4d, dataset, render_data, savedir, img_outpath,
                    specific_pose):

    # Render image
    with torch.no_grad():
        rgb, ref_images, target, scan = i4d.render_img(render_data,
                                                       dataset.render_factor,
                                                       dataset.H, dataset.W,
                                                       specific_pose)

        # Render the target
        if not specific_pose:
            filename = os.path.join(savedir, f'target.png')
            imageio.imwrite(filename, (target * 255).numpy().astype(np.uint8))

    # Save rendered image, converting to uint8
    # TODO: NOTE: added rgb * 255 here to fix warning about float values
    print("RGB Render", rgb.min(), rgb.max(), rgb.dtype)
    rgb = (rgb * 255).astype(np.uint8)
    imageio.imwrite(img_outpath, rgb)

    # Copy all reference images into rendering folder
    for i, ref_img in enumerate(ref_images):
        outpath = os.path.join(savedir, f'ref_img_{i}.png')
        if not os.path.exists(outpath):
            imageio.imwrite(outpath, (ref_img * 255).numpy().astype(np.uint8))

    # Put all reference images in a single image and save
    outpath = os.path.join(savedir, f'ref_images.png')
    if not os.path.exists(outpath):
        plt.figure(figsize=(50, 20), dpi=200)
        plt.xticks([]), plt.yticks([])
        for i in range(10):

            ax = plt.subplot(2, 5, i + 1)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            ax.imshow(ref_images[i], interpolation='bicubic')

        plt.savefig(outpath)
        plt.close()

    return rgb


def str2bool(x):
    if isinstance(x, bool):
        return x

    return x.lower() in ("true", "1")


if __name__ == '__main__':
    import config_loader
    import model
    from configargparse import DefaultConfigFileParser

    cfg = config_loader.get_config()
    cfg.video = True

    # Override architectural information from original experiment file
    orig_expname = cfg.expname.replace("render_", "")
    archi_file = os.path.join("configs", orig_expname + ".txt")
    print('archi_file', archi_file)

    if os.path.exists(archi_file):
        with open(archi_file, 'r') as f:
            orig_params = DefaultConfigFileParser().parse(f)
            print('orig_params', orig_params)
    else:
        print(
            f'WARNING: Could not find {archi_file}. Architectural options are not being loaded and thus can be incorrect.'
        )

    ## Override the original parameters
    cfg.num_transformer_layers = int(
        orig_params.get('num_transformer_layers', cfg.num_transformer_layers))
    cfg.num_attn_heads = int(
        orig_params.get('num_attn_heads', cfg.num_attn_heads))
    cfg.no_compression = str2bool(
        orig_params.get('no_compression', cfg.no_compression))
    cfg.reduce_features = str2bool(
        orig_params.get('reduce_features', cfg.reduce_features))

    print(
        "================Overriding the architectural parameters======================="
    )
    print('cfg.num_transformer_layers', cfg.num_transformer_layers)
    print('cfg.num_attn_heads', cfg.num_attn_heads)
    print('cfg.no_compression', cfg.no_compression)
    print('cfg.reduce_features', cfg.reduce_features)

    exit()

    # Generate/Render the images
    mode = 'test'
    dataset = SceneDataset(cfg, mode)
    i4d = model.Implicit4D(cfg, dataset.proj_pts_to_ref_torch)

    i4d.load_model()

    if cfg.dataset_type == 'DTU':
        for scan in cfg.generate_specific_samples:
            print('cfg.gen_pose', cfg.gen_pose)
            for pose_idx in cfg.gen_pose:
                pose = DTU.load_cam_path()[pose_idx]
                render_pose(cfg, i4d, dataset, i4d.start, scan,
                            (pose_idx, pose))