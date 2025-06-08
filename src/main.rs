// Global attributes to silence most warnings of "low" interest:
/*
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_unsafe)]
#![allow(unused_variables)]
*/
extern crate nalgebra_glm as glm;

use glm::Mat4;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::{mem, os::raw::c_void, ptr};

mod shader;
mod mesh;
mod scene_graph;
mod toolbox;
mod util;

use glutin::event::{DeviceEvent, ElementState::{Pressed, Released}, Event, KeyboardInput, VirtualKeyCode::{self, *}, WindowEvent};
use glutin::event_loop::ControlFlow;
use crate::scene_graph::SceneNode;

// initial window size
const INITIAL_SCREEN_W: u32 = 800;
const INITIAL_SCREEN_H: u32 = 600;


// Get the size of an arbitrary array of numbers measured in bytes
fn byte_size_of_array<T>(val: &[T]) -> isize {
    std::mem::size_of_val(&val[..]) as isize
}

// Get the OpenGL-compatible pointer to an arbitrary array of numbers
fn pointer_to_array<T>(val: &[T]) -> *const c_void {
    &val[0] as *const T as *const c_void
}

// Get the size of the given type in bytes
fn size_of<T>() -> i32 {
    mem::size_of::<T>() as i32
}

// Get an offset in bytes for n units of type T, represented as a relative pointer
fn offset<T>(n: u32) -> *const c_void {
    (n * mem::size_of::<T>() as u32) as *const T as *const c_void
}


/// Generates a VAO (Vertex Array Object) and fills it with data.
/// 1. Generate a VAO and bind it
/// 2. Generate a VBO and bind it
/// 3. Fill it with data
/// 4. Configure a VAP for the data and enable it
/// 5. Generate a IBO and bind it
/// 6. Fill it with data
/// 7. Return the ID of the VAO
unsafe fn create_vao(vertices: &Vec<f32>, indices: &Vec<u32>, colors: &Vec<f32>, normals: &Vec<f32>) -> u32 {
    let mut vao: u32 = 0;
    gl::GenVertexArrays(1, &mut vao);
    gl::BindVertexArray(vao);

    // Vertices
    let mut vbo: u32 = 0;
    gl::GenBuffers(1, &mut vbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(vertices),
        pointer_to_array(vertices),
        gl::STATIC_DRAW,
    );
    gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 3 * size_of::<f32>(), ptr::null());
    gl::EnableVertexAttribArray(0);

    // Colors
    let mut color_vbo: u32 = 0;
    gl::GenBuffers(1, &mut color_vbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, color_vbo);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(colors),
        pointer_to_array(colors),
        gl::STATIC_DRAW,
    );
    gl::VertexAttribPointer(1, 4, gl::FLOAT, gl::FALSE, 4 * size_of::<f32>(), ptr::null());
    gl::EnableVertexAttribArray(1);

    // Normals
    let mut normal_vbo: u32 = 0;
    gl::GenBuffers(1, &mut normal_vbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, normal_vbo);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(normals),
        pointer_to_array(normals),
        gl::STATIC_DRAW,
    );
    gl::VertexAttribPointer(2, 3, gl::FLOAT, gl::FALSE, 3 * size_of::<f32>(), ptr::null());
    gl::EnableVertexAttribArray(2);

    // Indices
    let mut ibo: u32 = 0;
    gl::GenBuffers(1, &mut ibo);
    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ibo);
    gl::BufferData(
        gl::ELEMENT_ARRAY_BUFFER,
        byte_size_of_array(indices),
        pointer_to_array(indices),
        gl::STATIC_DRAW,
    );

    gl::BindVertexArray(0); // Unbind the VAO

    return vao;
}

unsafe fn draw_scene(node: &SceneNode, view_projection_matrix: &glm::Mat4, transformation_so_far: &glm::Mat4, shader: &shader::Shader) {
    // Step 1: Apply local transformations (translation, rotation, scaling) to the node
    let translation = glm::translate(&glm::identity(), &node.position);
    let translation_to_origin = glm::translate(&glm::identity(), &-node.reference_point);
    let rotation =
        glm::rotation(node.rotation.x, &glm::vec3(1.0, 0.0, 0.0))
            * glm::rotation(node.rotation.y, &glm::vec3(0.0, 1.0, 0.0))
            * glm::rotation(node.rotation.z, &glm::vec3(0.0, 0.0, 1.0));
    let translation_back = glm::translate(&glm::identity(), &node.reference_point);
    let scale = glm::scaling(&node.scale);

    // Combine local transformations into the local transformation matrix
    let local_transformation = translation * translation_back * rotation * translation_to_origin * scale;

    // Step 2: Combine the parent’s transformation with the current node’s local transformation
    let model_matrix = transformation_so_far * local_transformation;

    // Compute the normal matrix (3x3, based on rotation part of model matrix)
    let normal_matrix = glm::mat4_to_mat3(&glm::transpose(&glm::inverse(&model_matrix)));

    // Step 3: Combine the model matrix with the view projection matrix to get the MVP matrix
    let mvp_matrix = view_projection_matrix * model_matrix;

    // Step 4: Send the MVP and the normal matrix to the shader
    let transform_location = shader.get_uniform_location("transform");
    gl::UniformMatrix4fv(transform_location, 1, gl::FALSE, mvp_matrix.as_ptr());

    let normal_matrix_location = shader.get_uniform_location("normalMatrix");
    gl::UniformMatrix3fv(normal_matrix_location, 1, gl::FALSE, normal_matrix.as_ptr());

    // Step 5: If this node has a VAO, it's drawable so bind it and draw it
    if node.vao_id != 0 {
        gl::BindVertexArray(node.vao_id);
        gl::DrawElements(
            gl::TRIANGLES,
            node.index_count,
            gl::UNSIGNED_INT,
            ptr::null(),
        );
        gl::BindVertexArray(0);
    }

    // Step 6: Recursively draw each child node
    for &child in &node.children {
        draw_scene(&*child, view_projection_matrix, &model_matrix, shader);
    }
}


fn main() {
    // Set up the necessary objects to deal with windows and event handling
    let el = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Gloom-rs")
        .with_resizable(true)
        .with_inner_size(glutin::dpi::LogicalSize::new(INITIAL_SCREEN_W, INITIAL_SCREEN_H));
    let cb = glutin::ContextBuilder::new()
        .with_vsync(true);
    let windowed_context = cb.build_windowed(wb, &el).unwrap();

    // Set up a shared vector for keeping track of currently pressed keys
    let arc_pressed_keys = Arc::new(Mutex::new(Vec::<VirtualKeyCode>::with_capacity(10)));
    // Make a reference of this vector to send to the render thread
    let pressed_keys = Arc::clone(&arc_pressed_keys);

    // Set up shared tuple for tracking mouse movement between frames
    let arc_mouse_delta = Arc::new(Mutex::new((0f32, 0f32)));
    // Make a reference of this tuple to send to the render thread
    let mouse_delta = Arc::clone(&arc_mouse_delta);

    // Set up shared tuple for tracking changes to the window size
    let arc_window_size = Arc::new(Mutex::new((INITIAL_SCREEN_W, INITIAL_SCREEN_H, false)));
    // Make a reference of this tuple to send to the render thread
    let window_size = Arc::clone(&arc_window_size);

    // Spawn a separate thread for rendering, so event handling doesn't block rendering
    let render_thread = thread::spawn(move || {
        // Acquire the OpenGL Context and load the function pointers.
        // This has to be done inside of the rendering thread, because
        // an active OpenGL context cannot safely traverse a thread boundary
        let context = unsafe {
            let c = windowed_context.make_current().unwrap();
            gl::load_with(|symbol| c.get_proc_address(symbol) as *const _);
            c
        };

        let mut window_aspect_ratio = INITIAL_SCREEN_W as f32 / INITIAL_SCREEN_H as f32;

        // Set up openGL
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);
            gl::Enable(gl::CULL_FACE);
            gl::Disable(gl::MULTISAMPLE);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::DebugMessageCallback(Some(util::debug_callback), ptr::null());

            // Print some diagnostics
            println!("{}: {}", util::get_gl_string(gl::VENDOR), util::get_gl_string(gl::RENDERER));
            println!("OpenGL\t: {}", util::get_gl_string(gl::VERSION));
            println!("GLSL\t: {}", util::get_gl_string(gl::SHADING_LANGUAGE_VERSION));
        }

        // Setting up the VAO

        let terrain_mesh = mesh::Terrain::load("resources/lunarsurface.obj");

        let helicopter = mesh::Helicopter::load("resources/helicopter.obj");


        // Define vertices for 3 overlapping triangles
        let vertices: Vec<f32> = vec![
            // Triangle 1 (Center), (z = -3.5) closest
            -0.5, -0.5, -3.5,  // Left
            0.5, -0.5, -3.5,  // Right
            0.0,  0.5, -3.5,  // Top

            // Triangle 2 (Shifted to the right), (z = -3.3) middle
            -0.25, -0.5, -3.3, // Left
            0.75, -0.5, -3.3, // Right
            0.25,  0.5, -3.3, // Top

            // Triangle 3 (Shifted to the left), (z = -3.1) farthest
            -0.75, -0.5, -3.1, // Left
            0.25, -0.5, -3.1, // Right
            -0.25,  0.5, -3.1, // Top
        ];

        // Define colors for each triangle (RGBA with transparency, alpha < 1)
        let colors: Vec<f32> = vec![
            // Colors for Triangle 1 (Red with transparency)
            1.0, 0.0, 0.0, 0.5,  // Bottom left
            1.0, 0.0, 0.0, 0.5,  // Bottom right
            1.0, 0.0, 0.0, 0.5,  // Top

            // Colors for Triangle 2 (Green with transparency)
            0.0, 1.0, 0.0, 0.5,  // Bottom left
            0.0, 1.0, 0.0, 0.5,  // Bottom right
            0.0, 1.0, 0.0, 0.5,  // Top

            // Colors for Triangle 3 (Blue with transparency)
            0.0, 0.0, 1.0, 0.5,  // Bottom left
            0.0, 0.0, 1.0, 0.5,  // Bottom right
            0.0, 0.0, 1.0, 0.5,  // Top
        ];

        // Define indices for the 3 triangles
        let indices: Vec<u32> = vec![
            0, 1, 2, // Triangle 1 (closest)
            3, 4, 5, // Triangle 2 (middle)
            6, 7, 8, // Triangle 3 (farthest)
        ];

        // Create the VAO
        let triangles_vao = unsafe { create_vao(&vertices, &indices, &colors, &vec![-1.0; vertices.len()]) };

        let terrain_vao = unsafe {
            create_vao(&terrain_mesh.vertices, &terrain_mesh.indices, &terrain_mesh.colors, &terrain_mesh.normals)
        };

        // Create a root node for the whole scene
        let mut scene_root_node = SceneNode::new();

        let mut terrain_node = SceneNode::from_vao(terrain_vao, terrain_mesh.index_count);

        // Create a node for the triangles
        let triangles_node = SceneNode::from_vao(triangles_vao, indices.len() as i32);

        // Add the terrain as a child to the root node of the scene
        scene_root_node.add_child(&terrain_node);

        // Add the triangles as a child to the root node of the scene
        scene_root_node.add_child(&triangles_node);

        // Setting the terrain's position (centered at the origin)
        terrain_node.position = glm::vec3(0.0, 0.0, 0.0);


        let helicopter_body_vao = unsafe {
            create_vao(&helicopter.body.vertices, &helicopter.body.indices, &helicopter.body.colors, &helicopter.body.normals)
        };

        let helicopter_door_vao = unsafe {
            create_vao(&helicopter.door.vertices, &helicopter.door.indices, &helicopter.door.colors, &helicopter.door.normals)
        };

        let helicopter_main_rotor_vao = unsafe {
            create_vao(&helicopter.main_rotor.vertices, &helicopter.main_rotor.indices, &helicopter.main_rotor.colors, &helicopter.main_rotor.normals)
        };

        let helicopter_tail_rotor_vao = unsafe {
            create_vao(&helicopter.tail_rotor.vertices, &helicopter.tail_rotor.indices, &helicopter.tail_rotor.colors, &helicopter.tail_rotor.normals)
        };


        let mut helicopters = vec![]; // Store multiple helicopters

        // Create 5 helicopters, each with a different initial position or offset
        for i in 0..5 {
            // Load helicopter parts
            let mut helicopter_body_node = SceneNode::from_vao(helicopter_body_vao, helicopter.body.index_count);
            let helicopter_door_node = SceneNode::from_vao(helicopter_door_vao, helicopter.door.index_count);
            let helicopter_main_rotor_node = SceneNode::from_vao(helicopter_main_rotor_vao, helicopter.main_rotor.index_count);
            let mut helicopter_tail_rotor_node = SceneNode::from_vao(helicopter_tail_rotor_vao, helicopter.tail_rotor.index_count);

            // The tail rotor rotates around its pivot point near the end of the tail
            helicopter_tail_rotor_node.reference_point = glm::vec3(0.35, 2.3, 10.4); // Specified in the task description

            // Create the root node for each helicopter
            let mut helicopter_root_node = SceneNode::new();

            // Attach parts to the helicopter
            helicopter_root_node.add_child(&helicopter_body_node);
            helicopter_body_node.add_child(&helicopter_door_node);
            helicopter_body_node.add_child(&helicopter_main_rotor_node);
            helicopter_body_node.add_child(&helicopter_tail_rotor_node);

            // Offset each helicopter's initial position to avoid collisions
            let offset = i as f32 * 20.0;  // Offset each helicopter by 20 units
            helicopter_root_node.position = glm::vec3(offset, 10.0, offset);

            // Add the helicopter to the scene
            scene_root_node.add_child(&helicopter_root_node);

            // Store helicopter nodes in the vector
            helicopters.push(helicopter_root_node);
        }


        // Setting up the shaders
        // The `.` in the path is relative to `Cargo.toml`.
        let simple_shader = unsafe {
            shader::ShaderBuilder::new()
                .attach_file("./shaders/simple.vert")
                .attach_file("./shaders/simple.frag")
                .link()
        };

        // Activate the shader program
        unsafe {
            simple_shader.activate();
        }


        // Camera position and rotation variables
        let mut camera_x: f32 = 0.0;
        let mut camera_y: f32 = 0.0;
        let mut camera_z: f32 = 0.0;
        let mut camera_yaw: f32 = -1.6; // Spawn right in front of the triangles
        let mut camera_pitch: f32 = 0.0;


        // Set up the projection matrix
        let fovy: f32 = 45.0_f32.to_radians(); // Field of View (in radians)
        let near: f32 = 1.0;   // Near clipping plane
        let far: f32 = 1000.0; // Far clipping plane

        // Controlling helicopter index
        let mut controlling_helicopter: Option<usize> = None;
        // Chase camera settings
        let chase_radius: f32 = 20.0;
        let min_distance: f32 = 5.0;   // Minimum allowed distance from the helicopter (buffer zone)
        let mut alt_pressed = false; // Alt key pressed, a flag to enable helicopter controls

        let projection_matrix: Mat4 = glm::perspective(window_aspect_ratio, fovy, near, far);


        // The main rendering loop
        let first_frame_time = std::time::Instant::now();
        let mut previous_frame_time = first_frame_time;
        loop {
            // Compute time passed since the previous frame and since the start of the program
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(first_frame_time).as_secs_f32();
            let delta_time = now.duration_since(previous_frame_time).as_secs_f32();
            previous_frame_time = now;

            // Handle resize events
            if let Ok(mut new_size) = window_size.lock() {
                if new_size.2 {
                    context.resize(glutin::dpi::PhysicalSize::new(new_size.0, new_size.1));
                    window_aspect_ratio = new_size.0 as f32 / new_size.1 as f32;
                    (*new_size).2 = false;
                    println!("Window was resized to {}x{}", new_size.0, new_size.1);
                    unsafe { gl::Viewport(0, 0, new_size.0 as i32, new_size.1 as i32); }
                }
            }

            // Calculate the camera's direction based on yaw and pitch
            let camera_direction = glm::vec3(
                camera_yaw.cos() * camera_pitch.cos(),  // X component (yaw and pitch)
                camera_pitch.sin(),                    // Y component (pitch)
                camera_yaw.sin() * camera_pitch.cos()   // Z component (yaw and pitch)
            ).normalize();

            // Handle keyboard input
            if let Ok(keys) = pressed_keys.lock() {

                for key in keys.iter() {

                    let ctrl_pressed = keys.contains(&VirtualKeyCode::LControl) || keys.contains(&VirtualKeyCode::RControl);

                    if ctrl_pressed && controlling_helicopter.is_some() {
                        controlling_helicopter = None;
                    }

                    if ctrl_pressed {
                        match key {
                            VirtualKeyCode::Key1 => controlling_helicopter = Some(0),
                            VirtualKeyCode::Key2 => controlling_helicopter = Some(1),
                            VirtualKeyCode::Key3 => controlling_helicopter = Some(2),
                            VirtualKeyCode::Key4 => controlling_helicopter = Some(3),
                            VirtualKeyCode::Key5 => controlling_helicopter = Some(4),
                            _ => {
                                controlling_helicopter = None;
                            }
                        }
                    }


                    // Control the selected helicopter
                    if controlling_helicopter.is_some() {
                        let controllable_helicopter = &mut helicopters[controlling_helicopter.unwrap()];

                        // Calculate the forward vector based on the helicopter's yaw (rotation.y)
                        let helicopter_direction = glm::vec3(
                            controllable_helicopter.rotation.y.cos() * controllable_helicopter.rotation.x.cos(),
                            controllable_helicopter.rotation.x.sin(),
                            controllable_helicopter.rotation.y.sin() * controllable_helicopter.rotation.x.cos()
                        ).normalize();

                        // Calculate the right vector: perpendicular to the helicopter direction and world up vector
                        let right = glm::cross(&glm::vec3(0.0, 1.0, 0.0), &helicopter_direction).normalize();

                        // Control helicopter movement based on key input
                        match key {
                            // Move forward in the direction the helicopter is facing
                            VirtualKeyCode::W => {
                                controllable_helicopter.position += helicopter_direction * delta_time * 5.0;
                            }
                            // Move backward
                            VirtualKeyCode::S => {
                                controllable_helicopter.position -= helicopter_direction * delta_time * 5.0;
                            }
                            // Strafe left
                            VirtualKeyCode::A => {
                                controllable_helicopter.position -= right * delta_time * 5.0;
                            }
                            // Strafe right
                            VirtualKeyCode::D => {
                                controllable_helicopter.position += right * delta_time * 5.0;
                            }
                            // Move up/down (Space for up, LShift for down)
                            VirtualKeyCode::Space => controllable_helicopter.position.y += delta_time * 5.0,
                            VirtualKeyCode::LShift => controllable_helicopter.position.y -= delta_time * 5.0,

                            // Adjust pitch (up/down tilt)
                            VirtualKeyCode::Up => controllable_helicopter.rotation.x += delta_time * 1.0,
                            VirtualKeyCode::Down => controllable_helicopter.rotation.x -= delta_time * 1.0,

                            // Adjust yaw (left/right rotation)
                            VirtualKeyCode::Left => controllable_helicopter.rotation.y -= delta_time * 1.0,
                            VirtualKeyCode::Right => controllable_helicopter.rotation.y += delta_time * 1.0,

                            _ => {}
                        }
                    }

                    // Right vector: perpendicular to the camera direction and the world up vector
                    let right = glm::cross(&camera_direction, &glm::vec3(0.0, 1.0, 0.0)).normalize();

                    // How much to move the camera per frame
                    let delta = 35.0;

                    match key {
                        // The `VirtualKeyCode` enum is defined here:
                        //    https://docs.rs/winit/0.25.0/winit/event/enum.VirtualKeyCode.html

                        // Camera movement (WASD + Space and LShift)
                        VirtualKeyCode::W => {
                            camera_x += camera_direction.x * delta_time * delta;
                            camera_y += camera_direction.y * delta_time * delta;
                            camera_z += camera_direction.z * delta_time * delta;
                        }
                        VirtualKeyCode::S => {
                            camera_x -= camera_direction.x * delta_time * delta;
                            camera_y -= camera_direction.y * delta_time * delta;
                            camera_z -= camera_direction.z * delta_time * delta;
                        }
                        VirtualKeyCode::A => {
                            camera_x -= right.x * delta_time * delta;
                            camera_y -= right.y * delta_time * delta;
                            camera_z -= right.z * delta_time * delta;
                        }
                        VirtualKeyCode::D => {
                            camera_x += right.x * delta_time * delta;
                            camera_y += right.y * delta_time * delta;
                            camera_z += right.z * delta_time * delta;
                        }
                        VirtualKeyCode::Space => camera_y += delta_time * delta,
                        VirtualKeyCode::LShift => camera_y -= delta_time * delta,

                        // Camera rotation (arrow keys)
                        VirtualKeyCode::Up => camera_pitch -= delta_time * 15.0,   // Pitch up
                        VirtualKeyCode::Down => camera_pitch += delta_time * 15.0, // Pitch down
                        VirtualKeyCode::Left => camera_yaw -= delta_time * 15.0,   // Yaw left
                        VirtualKeyCode::Right => camera_yaw += delta_time * 15.0,  // Yaw right

                        VirtualKeyCode::LAlt | VirtualKeyCode::RAlt => alt_pressed = !alt_pressed,

                        // default handler:
                        _ => {}
                    }

                    // Limit the pitch to prevent weird camera rotation
                    if camera_pitch > std::f32::consts::FRAC_PI_2 {
                        camera_pitch = std::f32::consts::FRAC_PI_2;
                    } else if camera_pitch < -std::f32::consts::FRAC_PI_2 {
                        camera_pitch = -std::f32::consts::FRAC_PI_2;
                    }
                }
            }
            // Handle mouse movement. delta contains the x and y movement of the mouse since last frame in pixels
            if let Ok(mut delta) = mouse_delta.lock() {
                // Optionally access the accumulated mouse movement between
                // frames here with `delta.0` and `delta.1`

                *delta = (0.0, 0.0); // reset when done
            }

            // Animate each helicopter with an offset to prevent collisions
            for (i, helicopter) in helicopters.iter_mut().enumerate() {
                if alt_pressed && controlling_helicopter.is_some() && controlling_helicopter.unwrap() == i {
                    // Rotate the main rotor
                    let helicopter_main_rotor_node = helicopter.get_child(0).get_child(1);
                    helicopter_main_rotor_node.rotation.y += delta_time * 2_070.0;

                    // Rotate the tail rotor
                    let helicopter_tail_rotor_node = helicopter.get_child(0).get_child(2);
                    helicopter_tail_rotor_node.rotation.x += delta_time * 2_070.0;

                    continue;
                }

                // Offset the time for each helicopter by its index
                let heading = toolbox::simple_heading_animation(elapsed + (i as f32) * 5.0);

                // Update the helicopter's position and rotation
                helicopter.position.x = heading.x;
                helicopter.position.z = heading.z;
                helicopter.rotation.y = heading.yaw;
                helicopter.rotation.x = heading.pitch;
                helicopter.rotation.z = heading.roll;

                // Rotate the main rotor
                let helicopter_main_rotor_node = helicopter.get_child(0).get_child(1);
                helicopter_main_rotor_node.rotation.y += delta_time * 2_070.0;

                // Rotate the tail rotor
                let helicopter_tail_rotor_node = helicopter.get_child(0).get_child(2);
                helicopter_tail_rotor_node.rotation.x += delta_time * 2_070.0;
            }



            // Desired camera angle (45 degrees above the helicopter)
            let angle_above = 45.0_f32.to_radians();
            let vertical_offset = chase_radius * angle_above.sin();  // Y offset based on 45 degrees
            let horizontal_offset = chase_radius * angle_above.cos();  // XZ offset


            let mut view_matrix = glm::identity();
            if let Some(helicopter_index) = controlling_helicopter {
                // Chase camera logic
                let controlled_helicopter = &helicopters[helicopter_index];
                let helicopter_position = controlled_helicopter.position;

                // Calculate the camera's target position, 45 degrees above and behind the helicopter
                let direction_behind_helicopter = glm::vec3(
                    controlled_helicopter.rotation.y.sin(),
                    0.0,  // Only use XZ plane for direction
                    -controlled_helicopter.rotation.y.cos()
                ).normalize();

                // Calculate the target camera position (behind and above the helicopter)
                let target_camera_position = helicopter_position
                    - direction_behind_helicopter * horizontal_offset  // Move behind the helicopter
                    + glm::vec3(0.0, vertical_offset, 0.0);  // Move up by the vertical offset

                // Calculate the current distance between the camera and the target position
                let camera_position = glm::vec3(camera_x, camera_y, camera_z);
                let distance = glm::distance(&camera_position, &target_camera_position);

                // If the camera is outside the chase radius, move the camera closer to the target position
                if distance > chase_radius {
                    let direction_to_target = (target_camera_position - camera_position).normalize();
                    camera_x += direction_to_target.x * (distance - chase_radius) * delta_time;
                    camera_y += direction_to_target.y * (distance - chase_radius) * delta_time;
                    camera_z += direction_to_target.z * (distance - chase_radius) * delta_time;
                }

                // Prevent the camera from getting too close to the helicopter
                if distance < min_distance {
                    let direction_away_from_helicopter = (camera_position - helicopter_position).normalize();
                    camera_x += direction_away_from_helicopter.x * (min_distance - distance) * delta_time;
                    camera_y += direction_away_from_helicopter.y * (min_distance - distance) * delta_time;
                    camera_z += direction_away_from_helicopter.z * (min_distance - distance) * delta_time;
                }

                // Ensure the camera is always looking at the helicopter
                view_matrix = glm::look_at(
                    &glm::vec3(camera_x, camera_y, camera_z),
                    &helicopter_position,  // Camera looks at the helicopter
                    &glm::vec3(0.0, 1.0, 0.0)  // World up vector
                );
            } else {
                // Free camera logic
                let camera_position = glm::vec3(camera_x, camera_y, camera_z);
                let up = glm::vec3(0.0, 1.0, 0.0);  // World up vector

                // Create the view matrix using the look_at function
                view_matrix = glm::look_at(
                    &camera_position,                             // Camera position
                    &(camera_position + camera_direction),  // Where the camera is looking (camera_position + direction)
                    &up                                           // World up vector (constraining tilt to the Y-axis)
                );
            }

            let view_projection_matrix = projection_matrix * view_matrix;

            // Get the location of the 'transform' uniform in the shader
            let transform_location = unsafe { simple_shader.get_uniform_location("transform") };

            // Send the final matrix to the shader
            unsafe {
                gl::UniformMatrix4fv(transform_location, 1, gl::FALSE, view_projection_matrix.as_ptr());
            }


            unsafe {
                // Clear the color and depth buffers
                gl::ClearColor(0.035, 0.046, 0.078, 1.0); // night sky
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                // Issue the necessary gl:: commands to draw the scene
                let now = std::time::Instant::now();
                let elapsed = now.duration_since(first_frame_time).as_secs_f32();
                let delta_time = now.duration_since(previous_frame_time).as_secs_f32();
                previous_frame_time = now;

                simple_shader.activate();

                draw_scene(&scene_root_node, &view_projection_matrix, &glm::identity(), &simple_shader);

                // Unbind the VAO
                gl::BindVertexArray(0);
            }

            // Display the new color buffer on the display
            context.swap_buffers().unwrap(); // we use "double buffering" to avoid artifacts
        }
    });


    // ----------------- //
    // Internals
    // ----------------- //


    // Keep track of the health of the rendering thread
    let render_thread_healthy = Arc::new(RwLock::new(true));
    let render_thread_watchdog = Arc::clone(&render_thread_healthy);
    thread::spawn(move || {
        if !render_thread.join().is_ok() {
            if let Ok(mut health) = render_thread_watchdog.write() {
                println!("Render thread panicked!");
                *health = false;
            }
        }
    });

    // Start the event loop -- This is where window events are initially handled
    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        // Terminate program if render thread panics
        if let Ok(health) = render_thread_healthy.read() {
            if *health == false {
                *control_flow = ControlFlow::Exit;
            }
        }

        match event {
            Event::WindowEvent { event: WindowEvent::Resized(physical_size), .. } => {
                println!("New window size received: {}x{}", physical_size.width, physical_size.height);
                if let Ok(mut new_size) = arc_window_size.lock() {
                    *new_size = (physical_size.width, physical_size.height, true);
                }
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            // Keep track of currently pressed keys to send to the rendering thread
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput {
                    input: KeyboardInput { state: key_state, virtual_keycode: Some(keycode), .. }, ..
                }, ..
            } => {
                if let Ok(mut keys) = arc_pressed_keys.lock() {
                    match key_state {
                        Released => {
                            if keys.contains(&keycode) {
                                let i = keys.iter().position(|&k| k == keycode).unwrap();
                                keys.remove(i);
                            }
                        }
                        Pressed => {
                            if !keys.contains(&keycode) {
                                keys.push(keycode);
                            }
                        }
                    }
                }

                // Handle Escape and Q keys separately
                match keycode {
                    Escape => { *control_flow = ControlFlow::Exit; }
                    Q => { *control_flow = ControlFlow::Exit; }
                    _ => {}
                }
            }
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                // Accumulate mouse movement
                if let Ok(mut position) = arc_mouse_delta.lock() {
                    *position = (position.0 + delta.0 as f32, position.1 + delta.1 as f32);
                }
            }
            _ => {}
        }
    });
}
