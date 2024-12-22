//const { add } = require('./build/Release/addon.node');
//D:\Demo\CVD_VAD\build\Release
const cvd_addon = require("C:\\Users\\ujjawalkumar_verific\\Documents\\Projects\\DMV\\CvAddon_x64\\AddonResponseApp\\Win_x64_10_10_2024_Release_v2.1.8\\Release\\CVD_addon.node");
console.log("fork object creation",cvd_addon);
const child_config={
    is_decrypted:false,
    is_init:false,
    frame_count:0
};

process.on('message', (message) => {
    
    if (message.n==='init'){
        console.log('Process message',message);
        console.log(child_config.is_decrypted);
        if(child_config.is_decrypted){
            const result = cvd_addon.init_module(message.Configuration, message.path_config);
            let n=message.n;
            process.send({n,result});
            if(result){
                child_config.is_init=true;
            }
        }
        
    }
    if (message.n==='decrypt'){
        console.log('Process message',message);
        if(!child_config.is_decrypted){
            const result = cvd_addon.decrypt_model(message.seven_z_dll_path,message.encypted_password,message.encrypted_model_path,message.decrypted_model_path);
            let n=message.n;
            process.send({n,result});
            if(result){
                child_config.is_decrypted=true;
            }
        }
        
    }
    if(message.n === 'setBaselineImg'){
        if(child_config.is_init){
            const result=cvd_addon.extract_Arc_OBFeatures(message.dirPath);
            let n=message.n;
            process.send({n,result});
        }
        
    }
    if (message.n==='VAD'){
        const result = cvd_addon.get_process_voice_sample_output(message.a);
        process.send(result);
    }
    if (message.n==='od'){
        const CV_RESULT = {
            '0': 'Mobile Camera',
            '1': 'Suspicious Mobile Activity',
            '2': 'Multiple people',
            '3': 'Suspicious Background Activity',
            '4': 'Left Session',
            '6': 'FSLA',
            '9': 'Left Screen',
            '22': 'Long Multiple people',
            '44': 'Long Left Session',
            '99': 'Long Left Screen',
            '-1': 'Camera Block',
            //'1000': 'Nothing',
            //'-1000': 'Exception',
            //'8': 'Normal Session After checking Imposter and FSLA',
        };
    
       
        if(child_config.is_init){

            let  imp_result=0;
            let Impviolation=null;
            let violation=null;
            child_config.frame_count=child_config.frame_count+1
            const result = cvd_addon.detectObject(message.frameBase64);
            violation = CV_RESULT[result.toString()] || null;
            if(child_config.frame_count % 7 === 0){
                imp_result=cvd_addon.detect_imposter(message.frameBase64);
                if (imp_result ===5){
                    violation='Imposter'
                }
            }
            let n=message.n;
            let frame_count=child_config.frame_count;
            let frameBase64=message.frameBase64;
            if(violation){
                process.send({n,frame_count,frameBase64,violation});
            }
            
        }
        
    }
});
