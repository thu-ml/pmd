#        # Train
#        for epoch in range(1, epoches + 1):
#            time_epoch = -time.time()
#            time_dist = 0
#            time_match = 0
#            time_sgd = 0
#
#            perm = np.random.permutation(n)
#            x_train = x_train[perm]
#            x_code = x_code[perm]
#
#            ws = []
#            losses = []
#            for t in range(n // match_batch_size):
#                x_batch = x_train[t*match_batch_size : (t+1)*match_batch_size]
#                code_batch = x_code[t*match_batch_size : (t+1)*match_batch_size]
#
#                # Generate latent variable
#                z = sess.run(z_op, feed_dict={batch_size_ph: match_batch_size})
#
#                # Compute distance
#                time_dist -= time.time()
#                if FLAGS.arch != 'ae':
#                    g_img = sess.run(generated_image, 
#                                     feed_dict={z_ph: z, is_training: False})
#                    obj_matrix = my_pw_distance(flatten(g_img), flatten(x_batch))
#                else:
#                    if dist != 'mmd':
#                        g_code = sess.run(generated_code,
#                                         feed_dict={z_ph: z, is_training: False})
#                        obj_matrix = my_pw_distance(g_code, code_batch)
#                time_dist += time.time()
#
#                # Compute minimal weight matching
#                time_match -= time.time()
#                if dist == 'mmd':
#                    ws = 0
#                    assigned_x = x_batch
#                    assigned_code = code_batch
#                    match_result = 0
#                else:
#                    a, match_result = get_assignments(obj_matrix, FLAGS.match)
#                    assigned_x      = x_batch[a]
#                    assigned_code   = code_batch[a]
#                    ws.append(match_result)
#                time_match += time.time()
#                # print(match_result, time_match)
#
#                # Perform SGD
#                time_sgd -= time.time()
#                for sgd_iters in range(match_batch_size // optimize_batch_size):
#                    sgd_z_batch = z[sgd_iters * optimize_batch_size : (sgd_iters+1) * optimize_batch_size]
#                    sgd_x_batch = assigned_x[sgd_iters * optimize_batch_size : (sgd_iters+1) * optimize_batch_size]
#                    if FLAGS.arch == 'ae':
#                        sgd_x_batch = assigned_code[sgd_iters * optimize_batch_size : (sgd_iters+1) * optimize_batch_size]
#                    # Gradient descent
#                    _, l = sess.run([infer, matched_obj], 
#                                    feed_dict={z_ph: sgd_z_batch,
#                                               x_ph: sgd_x_batch,
#                                               batch_size_ph: optimize_batch_size,
#                                               learning_rate_ph: learning_rate * t0 / (t0 + epoch),
#                                               is_training: True})
#                    losses.append(l)
#                time_sgd += time.time()
#
#            time_epoch += time.time()
#
#            # Generate figures
#            if epoch % FLAGS.lag == 0:
#                match_result, match_result2 = generate_imgs(run_name, epoch)
#                print('Epoch {} (total {:.1f}, dist {:.1f}, match {:.1f}, sgd {:.1f} s): W distance = {} approx W distance = {}, loss = {}'.format(epoch, time_epoch, time_dist, time_match, time_sgd, match_result, match_result2, np.mean(losses)))
#


#    def generate_imgs(run_name, epoch):
#        z = sess.run(z_op, feed_dict={batch_size_ph: output_batch_size})
#        if FLAGS.arch != 'ae':
#            g_img = sess.run(generated_image, feed_dict={z_ph: z, is_training: False})
#            obj_matrix = my_pw_distance(flatten(sorted_x_train), flatten(g_img))
#        else:
#            g_code, g_img = sess.run([generated_code, generated_image],
#                    feed_dict={z_ph: z, is_training: False, ae.is_training: False})
#            obj_matrix = my_pw_distance(sorted_x_code, g_code)
#
#        g_img            = g_img.reshape(-1, n_xl, n_xl, n_channels)
#        a, match_result  = get_assignments(obj_matrix, 'e')
#        _, match_result2 = get_assignments(obj_matrix, FLAGS.match)
#        assigned_mx      = g_img[a]
#
#        # Interweave the imgs
#        all_imgs = np.reshape(np.hstack((sorted_x_train, assigned_mx)), 
#                              (-1, n_xl, n_xl, n_channels))
#        name = '{}/outfile_{}_{}.jpg'.format(run_name, epoch, match_result)
#        utils.save_image_collections(all_imgs,   name, scale_each=True, shape=(Fx, Fy))
#
#        name = '{}/images_{}_{}.jpg'.format(run_name, epoch, match_result)
#        utils.save_image_collections(g_img,      name, scale_each=True, shape=(Fx, Fy//2))
#
#        name = '{}/small_images_{}_{}.jpg'.format(run_name, epoch, match_result)
#        utils.save_image_collections(g_img[:50], name, scale_each=True, shape=(5, 10))
#
#        # Make interpolation
#        for run in range(10):
#            pts = 5
#            zs = sess.run(z_op, feed_dict={batch_size_ph: pts})
#            izs = np.zeros((pts * 20, n_z)).astype(np.float32)
#            cnt = 0
#            for i in range(pts):
#                s = zs[i]
#                t = zs[(i+1)%pts]
#                delta = (t-s) / 20
#                for j in range(20):
#                    izs[cnt] = s + delta*j
#                    cnt += 1
#
#            g_img = sess.run(generated_image, feed_dict={z_ph: izs, is_training: False, ae.is_training: False}).reshape(-1, n_xl, n_xl, n_channels)
#            name = '{}/interpolation{}_{}_run{}.jpg'.format(run_name, epoch, match_result, run)
#            utils.save_image_collections(g_img, name, scale_each=True, shape=(pts, 20))
#
#        return match_result, match_result2

        #generated_image = generator(z_ph, n_x, normalizer_params)
        #flattened_x     = layers.flatten(x_ph)
        #flattened_g     = layers.flatten(generated_image)


        #generated_code  = generator(z_ph, n_code, normalizer_params)
        #generated_image = ae.decoder(generated_code)
        #flattened_x     = x_ph
        #flattened_g     = generated_code
