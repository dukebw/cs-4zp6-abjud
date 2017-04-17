using Microsoft.AspNetCore.Mvc;
using System.Collections.Generic;
using System.Linq;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using TextToMotionWeb.Data;
using Microsoft.AspNetCore.Authorization;
using TextToMotionWeb.Models;
using Microsoft.AspNetCore.Identity;
using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;

namespace TextToMotionWeb.Controllers
{
    public class DashboardController:Controller
    {
        private readonly UserManager<ApplicationUser> manager;


        public DashboardController(UserManager<ApplicationUser> UserManager)
        {
            manager = UserManager;
        }

        public async Task<ApplicationUser> GetCurrentUser()
        {
            return await manager.GetUserAsync(HttpContext.User);
        }



        [Authorize]
        public async Task<IActionResult> Index()
        {
            var user = await GetCurrentUser();
            ViewData["user"] = user;
            return View();
        }

        [Authorize]
        //GET
        public async Task<IActionResult> ProcessImage()
        {
            var user = await GetCurrentUser();
            ViewData["user"] = user;
            return View();
        }

        [Authorize]
        public IActionResult ProcessVideo()
        {
            return View();
        }

        [Authorize]
        public IActionResult ProcessStream()
        {
            return View();
        }
    }
}
